from typing import Dict, Iterable, Any, Tuple, Union
import logging
import math
import time
import os
import datetime
import random
import numpy as np
from itertools import chain

import torch

from allennlp.common.tqdm import Tqdm
from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params
from allennlp.common.util import (dump_metrics, gpu_memory_mb, peak_memory_mb,
                                  lazy_groups_of)
from allennlp.data import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models import Model
from allennlp.nn import util as nn_util
from allennlp.training.trainer_base import TrainerBase
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.optimizers import Optimizer
from allennlp.training import util as training_util

from my_library.optimisation import GanOptimizer
from my_library.models.data_augmentation import Gan


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TrainerBase.register("gan-base")
class GanTrainer(TrainerBase):
    def __init__(self,
                 model: Gan,
                 optimizer: torch.optim.Optimizer,
                 train_dataset: Iterable[Instance],
                 noise_dataset: Iterable[Instance] = None,
                 data_iterator: DataIterator = None,
                 noise_iterator: DataIterator = None,
                 validation_dataset: Iterable[Instance] = None,
                 test_dataset: Iterable[Instance] = None,
                 validation_metric: str = "-loss",
                 serialization_dir: str = None,
                 num_epochs: int = 20,
                 batch_size: int = 16,
                 cuda_device: int = 0,
                 patience: int = 5,
                 num_serialized_models_to_keep: int = 1,
                 keep_serialized_model_every_num_seconds: int = None,
                 num_loop_discriminator: int = 10,
                 num_loop_generator: int = 10,
                 num_loop_classifier_on_real: int = 10,
                 num_loop_classifier_on_fake: int = 10,
                 ) -> None:
        super().__init__(serialization_dir, cuda_device)

        self.model = model

        self.optimizer = optimizer

        self.train_dataset = train_dataset
        self.noise_dataset = noise_dataset

        self.data_iterator = data_iterator
        self.noise_iterator = noise_iterator

        self._validation_dataset = validation_dataset
        self._test_dataset = test_dataset

        self._num_epochs = num_epochs
        self._batch_size = batch_size

        self._metric_tracker = MetricTracker(patience, validation_metric)
        self._validation_metric = validation_metric[1:]

        self._checkpointer = Checkpointer(serialization_dir,
                                          keep_serialized_model_every_num_seconds,
                                          num_serialized_models_to_keep)
        self._batch_num_total = 0
        self.num_loop_discriminator = num_loop_discriminator
        self.num_loop_generator = num_loop_generator
        self.num_loop_classifier_on_real = num_loop_classifier_on_real
        self.num_loop_classifier_on_fake = num_loop_classifier_on_fake

    def _train_epoch_discriminator(self):
        logger.info('### Training discriminator ###')

        d_loss = 0.0
        batches_this_loop = 0

        self.optimizer.stage = 'discriminator'

        data_iterator = self.data_iterator(self.train_dataset)
        noise_iterator = self.noise_iterator(self.noise_dataset)
        loop = Tqdm.tqdm(range(self.num_loop_discriminator), total=self.num_loop_discriminator)

        for _ in loop:
            batches_this_loop += 1

            self.optimizer.zero_grad()

            batch = next(data_iterator)
            noise = next(noise_iterator)

            batch = nn_util.move_to_device(batch, self._cuda_devices[0])
            noise = nn_util.move_to_device(noise, self._cuda_devices[0])

            # Real example
            features = self.model.feature_extractor(batch['text'])
            real_validity = self.model.discriminator(features, batch['label'])["output"]

            # Fake example
            fake_data = self.model.generator(noise["array"], noise["label"])["output"]
            fake_validity = self.model.discriminator(fake_data, noise["label"])["output"]

            d_error = -torch.mean(real_validity) + torch.mean(fake_validity)

            d_error.backward()

            d_loss += d_error.item()

            self.optimizer.step()

            # Update the description with the latest metrics
            loop.set_description(
                training_util.description_from_metrics({'d_loss': d_loss / batches_this_loop}),
                refresh=False
            )
        return {'d_loss': d_loss / batches_this_loop}

    def _train_epoch_generator(self):
        logger.info('### Training generator ###')

        g_loss = 0.0
        batches_this_loop = 0

        self.optimizer.stage = 'generator'
        noise_iterator = self.noise_iterator(self.noise_dataset)
        loop = Tqdm.tqdm(range(self.num_loop_generator))

        for _ in loop:
            batches_this_loop += 1

            self.optimizer.zero_grad()

            noise = next(noise_iterator)
            noise = nn_util.move_to_device(noise, self._cuda_devices[0])

            generated = self.model.generator(noise["array"],
                                             noise["label"],
                                             self.model.discriminator)
            # fake_data = generated["output"]
            fake_error = generated["loss"]
            fake_error.backward()

            # fake_mean += fake_data.mean()
            # fake_stdev += fake_data.std()

            g_loss += fake_error.mean().item()

            self.optimizer.step()

            # Update the description with the latest metrics
            loop.set_description(
                training_util.description_from_metrics({'g_loss': g_loss / batches_this_loop}),
                refresh=False
            )
        return {'g_loss': g_loss / batches_this_loop}

    def _train_epoch_classifier_on_real(self):
        logger.info('### Training classifier on real data ###')

        cls_loss = 0.
        batches_this_loop = 0

        self.optimizer.stage = 'classifier'

        data_iterator = self.data_iterator(self.train_dataset)
        loop = Tqdm.tqdm(range(self.num_loop_classifier_on_real))

        for _ in loop:
            batches_this_loop += 1

            self.optimizer.zero_grad()

            batch = next(data_iterator)
            batch = nn_util.move_to_device(batch, self._cuda_devices[0])

            features = self.model.feature_extractor(batch['text'])
            cls_error = self.model.classifier(features, batch['label'])['loss']
            cls_error.backward()

            cls_loss += cls_error.mean().item()

            self.optimizer.step()

            # Update the description with the latest metrics
            loop.set_description(
                training_util.description_from_metrics({'cls_loss_on_real': cls_loss / batches_this_loop}),
                refresh=False
            )
        return {'cls_loss_on_real': cls_loss / batches_this_loop}

    def _train_epoch_classifier_on_fake(self):
        logger.info('### Training classifier on fake data ###')

        cls_loss = 0.
        batches_this_loop = 0

        self.optimizer.stage = 'classifier'

        noise_iterator = self.noise_iterator(self.noise_dataset)
        loop = Tqdm.tqdm(range(self.num_loop_classifier_on_fake))

        for _ in loop:
            batches_this_loop += 1

            self.optimizer.zero_grad()

            noise = next(noise_iterator)
            noise = nn_util.move_to_device(noise, self._cuda_devices[0])

            generated = self.model.generator(noise["array"],
                                             noise["label"])['output']
            cls_error = self.model.classifier(generated, noise['label'])['loss']
            cls_error.backward()

            cls_loss += cls_error.mean().item()

            self.optimizer.step()

            # Update the description with the latest metrics
            loop.set_description(
                training_util.description_from_metrics({'cls_loss_on_fake': cls_loss / batches_this_loop}),
                refresh=False
            )
        if batches_this_loop:
            return {'cls_loss_on_fake': cls_loss / batches_this_loop}
        else:
            return {'cls_loss_on_fake': 0.}

    def _train_epoch(self, epoch: int) -> Dict[str, float]:

        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        peak_cpu_usage = peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        gpu_usage = []
        for gpu, memory in gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        self.model.train()

        # (1/4) First train the discriminator
        loss_dis = self._train_epoch_discriminator()

        # (2/4) Then, train the generator
        loss_gen = self._train_epoch_generator()

        # (3/4) Nex, train the classifier on real data
        loss_cls_on_real = self._train_epoch_classifier_on_real()

        # (4/4) Finally, train the classifier on generated fake data
        loss_cls_on_fake = None
        if epoch >= 5:
            loss_cls_on_fake = self._train_epoch_classifier_on_fake()

        # return the metrics, and reset metrics as the epoch ends
        metrics = self.model.get_metrics(reset=True)
        metrics.update(loss_dis)
        metrics.update(loss_gen)
        metrics.update(loss_cls_on_real)
        if loss_cls_on_fake:
            metrics.update(loss_cls_on_fake)
        return metrics

    def test(self) -> Dict[str, Any]:
        logger.info("### Testing ###")
        with torch.no_grad():
            self.model.eval()

            test_iterator = self.data_iterator(self._test_dataset, num_epochs=1, shuffle=False)
            # val_iterator = lazy_groups_of(val_iterator, 1)
            num_test_batches = math.ceil(self.data_iterator.get_num_batches(self._test_dataset))
            test_generator_tqdm = Tqdm.tqdm(test_iterator, total=num_test_batches)

            batches_this_epoch = 0
            test_loss = 0.
            test_metrics = {}

            for batch in test_generator_tqdm:
                # batch = batch[0]
                batch = nn_util.move_to_device(batch, self._cuda_devices[0])
                features = self.model.feature_extractor(batch['text'])
                cls_error = self.model.classifier(features, batch['label'])['loss']

                batches_this_epoch += 1
                test_loss += cls_error.mean().item()

                # Update the description with the latest metrics
                test_metrics = training_util.get_metrics(self.model, test_loss, batches_this_epoch)
                description = training_util.description_from_metrics(test_metrics)
                test_generator_tqdm.set_description(description, refresh=False)

            return test_metrics

    def train(self) -> Dict[str, Any]:

        logger.info("Beginning training.")

        val_metrics: Dict[str, float] = {}
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()

        metrics['best_epoch'] = self._metric_tracker.best_epoch
        for key, value in self._metric_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value

        # train over epochs
        for epoch in range(self._num_epochs):
            epoch_start_time = time.time()

            # train over one epoch
            train_metrics = self._train_epoch(epoch)

            # get peak of memory usage
            if 'cpu_memory_MB' in train_metrics:
                metrics['peak_cpu_memory_MB'] = max(metrics.get('peak_cpu_memory_MB', 0),
                                                    train_metrics['cpu_memory_MB'])
            for key, value in train_metrics.items():
                if key.startswith('gpu_'):
                    metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)

            # Validation
            if self._validation_dataset is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it,
                    # and reset the metrics as validation ends.
                    val_loss, num_batches = self._validation_loss()
                    val_metrics = training_util.get_metrics(self.model, val_loss, num_batches, reset=True)

                    # Check validation metric for early stopping
                    this_epoch_val_metric = val_metrics[self._validation_metric]
                    self._metric_tracker.add_metric(this_epoch_val_metric)

                    if self._metric_tracker.should_stop_early():
                        logger.info("Ran out of patience.  Stopping training.")
                        break

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            if self._metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics['best_epoch'] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

                self._metric_tracker.best_epoch_metrics = val_metrics

            if self._serialization_dir:
                dump_metrics(os.path.join(self._serialization_dir, f'metrics_epoch_{epoch}.json'), metrics)

            # if self._learning_rate_scheduler:
            #     self._learning_rate_scheduler.step(this_epoch_val_metric, epoch)
            # if self._momentum_scheduler:
            #     self._momentum_scheduler.step(this_epoch_val_metric, epoch)

            self._save_checkpoint(epoch)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * (self._num_epochs / float(epoch + 1) - 1)
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1

        # Load the best model state before returning
        best_model_state = self._checkpointer.best_model_state()
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return metrics

    def _validation_loss(self) -> Tuple[float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("### Validating ###")
        self.model.eval()

        val_iterator = self.data_iterator(self._validation_dataset, num_epochs=1, shuffle=False)
        # val_iterator = lazy_groups_of(val_iterator, 1)
        num_validation_batches = math.ceil(self.data_iterator.get_num_batches(self._validation_dataset))
        val_generator_tqdm = Tqdm.tqdm(val_iterator, total=num_validation_batches)

        batches_this_epoch = 0
        val_loss = 0.

        for batch in val_generator_tqdm:
            # batch = batch[0]
            batch = nn_util.move_to_device(batch, self._cuda_devices[0])
            features = self.model.feature_extractor(batch['text'])
            cls_error = self.model.classifier(features, batch['label'])['loss']

            batches_this_epoch += 1
            val_loss += cls_error.mean().item()

            # Update the description with the latest metrics
            val_metrics = training_util.get_metrics(self.model, val_loss, batches_this_epoch)
            description = training_util.description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

        return val_loss, batches_this_epoch

    def _save_checkpoint(self, epoch: Union[int, str]) -> None:
        """
        Saves a checkpoint of the model to self._serialization_dir.
        Is a no-op if self._serialization_dir is None.
        Parameters
        ----------
        epoch : Union[int, str], required.
            The epoch of training.  If the checkpoint is saved in the middle
            of an epoch, the parameter is a string with the epoch and timestamp.
        """
        # These are the training states we need to persist.
        training_states = {
            "metric_tracker": self._metric_tracker.state_dict(),
            # "optimizer": self.optimizer.state_dict(),
            "batch_num_total": self._batch_num_total
        }

        # If we have a learning rate or momentum scheduler, we should persist them too.
        # if self._learning_rate_scheduler is not None:
        #     training_states["learning_rate_scheduler"] = self._learning_rate_scheduler.state_dict()
        # if self._momentum_scheduler is not None:
        #     training_states["momentum_scheduler"] = self._momentum_scheduler.state_dict()

        self._checkpointer.save_checkpoint(
            model_state=self.model.state_dict(),
            epoch=epoch,
            training_states=training_states,
            is_best_so_far=self._metric_tracker.is_best_so_far())

        # Restore the original values for parameters so that training will not be affected.
        # if self._moving_average is not None:
        #     self._moving_average.restore()

    @classmethod
    def from_params(cls,  # type: ignore
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False) -> 'GanTrainer':

        training_file = params.pop('training_file')
        dev_file = params.pop('dev_file')
        test_file = params.pop('test_file')

        config_file = params.pop('config_file')
        cuda_device = params.pop_int("cuda_device")

        # Data reader
        vocab_reader = DatasetReader.from_params(params.pop('vocab_reader'))
        train_reader = DatasetReader.from_params(params.pop('train_reader'))
        val_reader = DatasetReader.from_params(params.pop("val_reader"))

        vocab_dataset = vocab_reader.read(cached_path(training_file))
        train_dataset = train_reader.read(cached_path(training_file))
        validation_dataset = val_reader.read(cached_path(dev_file))
        test_dataset = val_reader.read(cached_path(test_file))

        noise_reader = DatasetReader.from_params(params.pop("noise_reader"))
        noise_dataset = noise_reader.read("")

        # Vocabulary
        vocab = Vocabulary.from_instances(vocab_dataset,
                                          # min_count={'tokens': 2},
                                          only_include_pretrained_words=True,
                                          max_vocab_size=config_file.max_vocab_size,
                                          )
        logging.info('Vocab size: %s' % vocab.get_vocab_size())
        # vocab.print_statistics()
        # print(vocab.get_token_to_index_vocabulary('labels'))
        # Iterator
        data_iterator = DataIterator.from_params(params.pop("training_iterator"))
        data_iterator.index_with(vocab)
        noise_iterator = DataIterator.from_params(params.pop("noise_iterator"))
        noise_iterator.index_with(vocab)

        # Model
        feature_extractor = Model.from_params(params.pop("feature_extractor"), vocab=vocab)
        generator = Model.from_params(params.pop("generator"), vocab=None)
        discriminator = Model.from_params(params.pop("discriminator"), vocab=None)
        classifier = Model.from_params(params.pop("classifier"))

        if isinstance(cuda_device, list):
            model_device = cuda_device[0]
        else:
            model_device = cuda_device
        if model_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            feature_extractor = feature_extractor.cuda(model_device)
            generator = generator.cuda(model_device)
            discriminator = discriminator.cuda(model_device)
            classifier = classifier.cuda(model_device)

        model = Gan(
            feature_extractor=feature_extractor,
            generator=generator,
            discriminator=discriminator,
            classifier=classifier
        )

        # Optimize
        parameters = [[n, p] for n, p in model.feature_extractor.named_parameters() if p.requires_grad] + \
                     [[n, p] for n, p in model.generator.named_parameters() if p.requires_grad] + \
                     [[n, p] for n, p in model.discriminator.named_parameters() if p.requires_grad] + \
                     [[n, p] for n, p in model.classifier.named_parameters() if p.requires_grad]
        optimizer = GanOptimizer.from_params(parameters, params.pop("optimizer"))

        # training_util.move_optimizer_to_cuda(generator_optimizer)
        # training_util.move_optimizer_to_cuda(discriminator_optimizer)

        num_epochs = params.pop_int("num_epochs")
        batch_size = params.pop_int("batch_size")
        patience = params.pop_int("patience")
        num_loop_discriminator = params.pop_int("num_loop_discriminator")
        num_loop_generator = params.pop_int("num_loop_generator")
        num_loop_classifier_on_real = params.pop_int("num_loop_classifier_on_real")
        num_loop_classifier_on_fake = params.pop_int("num_loop_classifier_on_fake")

        params.pop("trainer")

        params.assert_empty(__name__)

        return cls(model=model,
                   optimizer=optimizer,
                   train_dataset=train_dataset,
                   noise_dataset=noise_dataset,
                   data_iterator=data_iterator,
                   noise_iterator=noise_iterator,
                   validation_dataset=validation_dataset,
                   test_dataset=test_dataset,
                   serialization_dir=serialization_dir,
                   num_epochs=num_epochs,
                   batch_size=batch_size,
                   cuda_device=cuda_device,
                   patience=patience,
                   num_loop_discriminator=num_loop_discriminator,
                   num_loop_generator=num_loop_generator,
                   num_loop_classifier_on_real=num_loop_classifier_on_real,
                   num_loop_classifier_on_fake=num_loop_classifier_on_fake)
