from typing import Dict, Iterable, Any, Tuple
import logging
import math
import time
import datetime
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
                 optimizer: GanOptimizer,
                 train_dataset: Iterable[Instance],
                 noise_dataset: Iterable[Instance] = None,
                 data_iterator: DataIterator = None,
                 noise_iterator: DataIterator = None,
                 validation_dataset: Iterable[Instance] = None,
                 validation_metric: str = "-loss",
                 serialization_dir: str = None,
                 num_epochs: int = 20,
                 batch_size: int = 16,
                 cuda_device: int = 0,
                 patience: int = 5,
                 num_serialized_models_to_keep: int = 1,
                 keep_serialized_model_every_num_seconds: int = None,
                 ) -> None:
        super().__init__(serialization_dir, cuda_device)

        self.model = model

        self.optimizer = optimizer

        self.train_dataset = train_dataset
        self.noise_dataset = noise_dataset

        self.data_iterator = data_iterator
        self.noise_iterator = noise_iterator

        self._validation_dataset = validation_dataset

        self._num_epochs = num_epochs
        self._batch_size = batch_size

        self._metric_tracker = MetricTracker(patience, validation_metric)
        self._validation_metric = validation_metric[1:]

        self._checkpointer = Checkpointer(serialization_dir,
                                          keep_serialized_model_every_num_seconds,
                                          num_serialized_models_to_keep)
        self._batch_num_total = 0

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:

        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        peak_cpu_usage = peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        gpu_usage = []
        for gpu, memory in gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        self.model.train()

        generator_loss = 0.0
        discriminator_real_loss = 0.0
        discriminator_fake_loss = 0.0
        fake_mean = 0.0
        fake_stdev = 0.0

        # (1/3) First train the discriminator
        self.optimizer.stage = 'discriminator'

        data_iterator = self.data_iterator(self.train_dataset)
        noise_iterator = self.noise_iterator(self.noise_dataset)

        for _ in range(10):
            self.optimizer.zero_grad()

            batch = next(data_iterator)
            noise = next(noise_iterator)

            batch = nn_util.move_to_device(batch, self._cuda_devices[0])
            noise = nn_util.move_to_device(noise, self._cuda_devices[0])

            # extract features
            features = self.model.feature_extractor(batch['text'])

            # Real example, want discriminator to predict 1.
            ones = nn_util.move_to_device(torch.ones((self._batch_size, 1)), self._cuda_devices[0])
            real_error = self.model.discriminator(features, ones)["loss"]
            real_error.backward()

            # Fake example, want discriminator to predict 0.
            fake_data = self.model.generator(noise["array"])["output"]
            zeros = nn_util.move_to_device(torch.zeros((self._batch_size, 1)), self._cuda_devices[0])
            fake_error = self.model.discriminator(fake_data, zeros)["loss"]
            fake_error.backward()

            discriminator_real_loss += real_error.mean().item()
            discriminator_fake_loss += fake_error.mean().item()

            self.optimizer.step()

        # (2/3) Then, train the generator
        self.optimizer.stage = 'generator'
        for _ in range(10):
            self.optimizer.zero_grad()

            noise = next(noise_iterator)
            noise = nn_util.move_to_device(noise, self._cuda_devices[0])

            generated = self.model.generator(noise["array"], self.model.discriminator)
            fake_data = generated["output"]
            fake_error = generated["loss"]
            fake_error.backward()

            fake_mean += fake_data.mean()
            fake_stdev += fake_data.std()

            generator_loss += fake_error.mean().item()

            self.optimizer.step()

        # (3/3) Finally, train the classifier
        self.optimizer.stage = 'classifier'
        for _ in range(10):
            self.optimizer.zero_grad()

        return {
            "generator_loss": generator_loss,
            "discriminator_fake_loss": discriminator_fake_loss,
            "discriminator_real_loss": discriminator_real_loss,
            "mean": fake_mean / 100,
            "stdev": fake_stdev / 100
        }

    def train(self) -> Dict[str, Any]:

        logger.info("Beginning training.")

        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
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
            train_metrics = self.train_one_epoch(epoch)

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
                    # We have a validation set, so compute all the metrics on it.
                    val_loss, num_batches = self._validation_loss()
                    val_metrics = training_util.get_metrics(self.model, val_loss, num_batches, reset=True)

                    # Check validation metric for early stopping
                    this_epoch_val_metric = val_metrics[self._validation_metric]
                    self._metric_tracker.add_metric(this_epoch_val_metric)

                    if self._metric_tracker.should_stop_early():
                        logger.info("Ran out of patience.  Stopping training.")
                        break

            description = (f'gl: {metrics["generator_loss"]:.3f} '
                           f'dfl: {metrics["discriminator_fake_loss"]:.3f} '
                           f'drl: {metrics["discriminator_real_loss"]:.3f} '
                           f'mean: {metrics["mean"]:.2f} '
                           f'std: {metrics["stdev"]:.2f} ')
        return metrics

    def _validation_loss(self) -> Tuple[float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")
        self.model.eval()

        val_iterator = self.data_iterator(self._validation_dataset, num_epochs=1, shuffle=False)
        val_iterator = lazy_groups_of(val_iterator, 1)
        num_validation_batches = math.ceil(self.data_iterator.get_num_batches(self._validation_dataset))

        batches_this_epoch = 0
        val_loss = 0

        for batch in Tqdm.tqdm(val_iterator, total=num_validation_batches):
            pass

        return val_loss, batches_this_epoch


    @classmethod
    def from_params(cls,  # type: ignore
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False) -> 'GanTrainer':

        config_file = params.pop('config_file')
        training_file = params.pop('training_file')
        dev_file = params.pop('dev_file')
        test_file = params.pop('test_file')
        cuda_device = params.pop_int("cuda_device")

        # Data reader
        train_reader = DatasetReader.from_params(params.pop('train_reader'))
        val_reader = DatasetReader.from_params(params.pop("val_reader"))
        train_dataset = train_reader.read(cached_path(training_file))
        validation_dataset = val_reader.read(cached_path(dev_file))

        noise_reader = DatasetReader.from_params(params.pop("noise_reader"))
        noise_dataset = noise_reader.read("")

        # Vocabulary
        vocab = Vocabulary.from_instances(train_dataset,
                                          # min_count={'tokens': 2},
                                          only_include_pretrained_words=True,
                                          max_vocab_size=config_file.max_vocab_size,
                                          )
        logging.info('Vocab size: %s' % vocab.get_vocab_size())

        # Iterator
        data_iterator = DataIterator.from_params(params.pop("training_iterator"))
        data_iterator.index_with(vocab)
        noise_iterator = DataIterator.from_params(params.pop("noise_iterator"))

        # Model
        feature_extractor = Model.from_params(params.pop("feature_extractor"), vocab=vocab)
        generator = Model.from_params(params.pop("generator"))
        discriminator = Model.from_params(params.pop("discriminator"))
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

        params.pop("trainer")

        params.assert_empty(__name__)

        return cls(model=model,
                   optimizer=optimizer,
                   train_dataset=train_dataset,
                   noise_dataset=noise_dataset,
                   data_iterator=data_iterator,
                   noise_iterator=noise_iterator,
                   validation_dataset=validation_dataset,
                   serialization_dir=serialization_dir,
                   num_epochs=num_epochs,
                   batch_size=batch_size,
                   cuda_device=cuda_device,
                   patience=patience)
