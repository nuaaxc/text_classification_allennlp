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


def compute_gradient_penalty(D, real_samples, fake_samples, labels, cuda_device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, dtype=torch.float32).cuda(cuda_device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    fake = torch.ones((real_samples.size(0), 1), requires_grad=False).cuda(cuda_device)
    d_interpolates = D(interpolates, labels)["output"]
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


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
                 clip_value: float = 1,
                 no_gen: bool=False,
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

        self._val_loss_tracker = MetricTracker(patience, validation_metric)
        self._g_loss_tracker = MetricTracker(3, validation_metric)
        self._validation_metric = validation_metric[1:]

        self._checkpointer = Checkpointer(serialization_dir,
                                          keep_serialized_model_every_num_seconds,
                                          num_serialized_models_to_keep)
        self._batch_num_total = 0
        self.num_loop_discriminator = num_loop_discriminator
        self.num_loop_generator = num_loop_generator
        self.num_loop_classifier_on_real = num_loop_classifier_on_real
        self.num_loop_classifier_on_fake = num_loop_classifier_on_fake

        self.cuda_device = cuda_device
        self.clip_value = clip_value
        self.no_gen = no_gen

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

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(self.model.discriminator,
                                                        features.data,
                                                        fake_data.data,
                                                        torch.randint_like(noise["label"], 0, 3),
                                                        self.cuda_device)

            d_error = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty

            d_error.backward()
            self.optimizer.step()

            # Clip weights of discriminator
            for p in self.model.discriminator.parameters():
                p.data.clamp_(-self.clip_value, self.clip_value)

            d_loss += d_error.item()

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

        g_data = []

        for _ in loop:
            batches_this_loop += 1

            self.optimizer.zero_grad()

            noise = next(noise_iterator)
            noise = nn_util.move_to_device(noise, self._cuda_devices[0])

            generated = self.model.generator(noise["array"],
                                             noise["label"],
                                             self.model.discriminator)

            g_data.append(generated["output"].data.cpu().numpy())

            fake_error = generated["loss"]
            fake_error.backward()
            self.optimizer.step()

            g_loss += fake_error.item()

            # Update the description with the latest metrics
            loop.set_description(
                training_util.description_from_metrics({'g_loss': g_loss / batches_this_loop}),
                refresh=False
            )
        return {'g_loss': g_loss / batches_this_loop}, np.vstack(g_data)

    def _train_epoch_classifier_on_real(self):
        logger.info('### Training classifier on real data ###')

        cls_loss = 0.
        batches_this_loop = 0

        self.optimizer.stage = 'classifier'

        data_iterator = self.data_iterator(self.train_dataset)
        loop = Tqdm.tqdm(range(self.num_loop_classifier_on_real))

        r_data = []

        for _ in loop:
            batches_this_loop += 1

            self.optimizer.zero_grad()

            batch = next(data_iterator)
            batch = nn_util.move_to_device(batch, self._cuda_devices[0])

            features = self.model.feature_extractor(batch['text'])

            r_data.append(features.data.cpu().numpy())

            cls_error = self.model.classifier(features, batch['label'])['loss']
            cls_error.backward()

            cls_loss += cls_error.mean().item()

            self.optimizer.step()

            # Update the description with the latest metrics
            loop.set_description(
                training_util.description_from_metrics({'cls_loss_on_real': cls_loss / batches_this_loop}),
                refresh=False
            )
        return {'cls_loss_on_real': cls_loss / batches_this_loop}, np.vstack(r_data)

    def _train_epoch_classifier_on_fake(self):
        logger.info('### Training classifier on fake data ###')

        cls_loss = 0.
        batches_this_loop = 0

        self.optimizer.stage = 'classifier'

        # freeze feature_extractor
        for p in self.model.feature_extractor.parameters():
            p.requires_grad = False

        noise_iterator = self.noise_iterator(self.noise_dataset)
        loop = Tqdm.tqdm(range(self.num_loop_classifier_on_fake))

        g_data = []

        for _ in loop:
            batches_this_loop += 1

            self.optimizer.zero_grad()

            noise = next(noise_iterator)
            noise = nn_util.move_to_device(noise, self._cuda_devices[0])

            generated = self.model.generator(noise["array"],
                                             noise["label"])['output']

            g_data.append(generated.data.cpu().numpy())

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
            return {'cls_loss_on_fake': cls_loss / batches_this_loop}, np.vstack(g_data)
        else:
            return {'cls_loss_on_fake': 0.}, np.vstack(g_data)

    def _train_gan(self) -> Any:
        loss_d = self._train_epoch_discriminator()
        loss_g, g_data = self._train_epoch_generator()
        return loss_d, loss_g, g_data

    def _train_epoch(self, epoch: int, train_phase: str) -> Any:

        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        logger.info("Training phase: %s ..." % train_phase)
        self.model.train()

        metrics = {}

        if train_phase == 'gan':              # train the GAN
            loss_d, loss_g, g_data = self._train_gan()
            metrics.update(loss_d)
            metrics.update(loss_g)
            metrics.update({'g_data': g_data})

        elif train_phase == 'cls_on_real':      # train the classifier on real data
            loss_cls_on_real, r_data = self._train_epoch_classifier_on_real()
            metrics.update(self.model.get_metrics(reset=True))
            metrics.update(loss_cls_on_real)
            metrics.update({'r_data': r_data})

        elif train_phase == 'cls_on_fake':      # train the classifier on fake data
            loss_cls_on_fake, g_data = self._train_epoch_classifier_on_fake()
            metrics.update(loss_cls_on_fake)
            metrics.update({'g_data': g_data})

        else:
            raise ValueError('unknown training phase name %s.' % train_phase)

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

    def train(self) -> Any:

        logger.info("Beginning training.")

        val_metrics: Dict[str, float] = {}
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()

        metrics['best_epoch'] = self._val_loss_tracker.best_epoch
        for key, value in self._val_loss_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value

        r_data_epochs = {}
        g_data_epochs = {}
        d_loss_epochs = {}
        g_loss_epochs = {}
        cls_loss_on_real_epochs = {}

        train_phase = 'cls_on_real'

        # train (over epochs)
        for epoch in range(self._num_epochs):

            if epoch > 450:
                train_phase = 'cls_on_fake'

            epoch_start_time = time.time()

            # train (over one epoch)
            train_metrics = self._train_epoch(epoch, train_phase)

            if train_phase == 'cls_on_real':
                r_data_epochs[epoch] = train_metrics['r_data']
                cls_loss_on_real_epochs[epoch] = train_metrics['cls_loss_on_real']

            elif train_phase == 'gan':
                d_loss_epochs[epoch] = train_metrics['d_loss']
                g_loss_epochs[epoch] = train_metrics['g_loss']
                g_data_epochs[epoch] = train_metrics['g_data']

            elif train_phase == 'cls_on_fake':
                g_data_epochs[epoch] = train_metrics['g_data']

            else:
                raise ValueError('unknown training phase %s.' % train_phase)

            # Validation
            if self._validation_dataset is not None and 'cls' in train_phase:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it,
                    # and reset the metrics as validation ends.
                    val_loss, num_batches = self._validation_loss()
                    val_metrics = training_util.get_metrics(self.model, val_loss, num_batches, reset=True)

                    # Check validation metric for early stopping
                    this_epoch_val_metric = val_metrics[self._validation_metric]
                    self._val_loss_tracker.add_metric(this_epoch_val_metric)

                    # stop cls_on_real phase
                    if self._val_loss_tracker.should_stop_early() and 'real' in train_phase:
                        logger.info("Ran out of patience. Stopping training the classifier on real data.")
                        if self.no_gen:
                            break
                        # load best model
                        best_model_state = self._checkpointer.best_model_state()
                        if best_model_state:
                            self.model.load_state_dict(best_model_state)
                        # move to the next phase (gan)
                        train_phase = 'gan'
                        # reset metric tracker
                        self._val_loss_tracker.clear()
                        continue
                    # stop cls_on_real phase
                    elif self._val_loss_tracker.should_stop_early() and 'fake' in train_phase:
                        logger.info("Ran out of patience. Stopping training the classifier on fake data.")
                        # load best model
                        best_model_state = self._checkpointer.best_model_state()
                        if best_model_state:
                            self.model.load_state_dict(best_model_state)
                        break

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                if key != 'r_data' and key != 'g_data':
                    metrics["training_" + key] = value

            if 'cls' in train_phase:

                for key, value in val_metrics.items():
                    metrics["validation_" + key] = value

                if self._val_loss_tracker.is_best_so_far():
                    metrics['best_epoch'] = epoch
                    for key, value in val_metrics.items():
                        metrics["best_validation_" + key] = value

                    self._val_loss_tracker.best_epoch_metrics = val_metrics

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
        # best_model_state = self._checkpointer.best_model_state()
        # if best_model_state:
        #     self.model.load_state_dict(best_model_state)

        meta_data = {
            'r_data_epochs': r_data_epochs,
            'g_data_epochs': g_data_epochs,
            'd_loss_epochs': d_loss_epochs,
            'g_loss_epochs': g_loss_epochs,
            'cls_loss_on_real_epochs': cls_loss_on_real_epochs
        }
        return metrics, meta_data

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
            "metric_tracker": self._val_loss_tracker.state_dict(),
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
            is_best_so_far=self._val_loss_tracker.is_best_so_far())

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
        clip_value = params.pop_int("clip_value")
        no_gen = params.pop_int("no_gen")

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
                   num_loop_classifier_on_fake=num_loop_classifier_on_fake,
                   clip_value=clip_value,
                   no_gen=no_gen)
