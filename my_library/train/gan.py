from typing import Dict, Iterable, Any
import logging
import tqdm
import torch
import numpy as np
from itertools import chain

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
from allennlp.training.optimizers import Optimizer
from allennlp.training import util as training_util

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TrainerBase.register("gan-base")
class GanTrainer(TrainerBase):
    def __init__(self,
                 serialization_dir: str,
                 train_dataset: Iterable[Instance],
                 noise: Iterable[Instance],
                 feature_extractor: Model,
                 generator: Model,
                 discriminator: Model,
                 train_iterator: DataIterator,
                 noise_iterator: DataIterator,
                 generator_optimizer: torch.optim.Optimizer,
                 discriminator_optimizer: torch.optim.Optimizer,
                 batches_per_epoch: int,
                 num_epochs: int,
                 batch_size: int,
                 cuda_device: int) -> None:
        super().__init__(serialization_dir, cuda_device)
        self.train_dataset = train_dataset
        self.noise = noise
        self.feature_extractor = feature_extractor
        self.generator = generator
        self.generator_optimizer = generator_optimizer
        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.num_epochs = num_epochs
        self.train_iterator = train_iterator
        self.noise_iterator = noise_iterator
        self.batches_per_epoch = batches_per_epoch
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:

        logger.info("Epoch %d/%d", epoch, self.num_epochs - 1)
        peak_cpu_usage = peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        gpu_usage = []
        for gpu, memory in gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        self.generator.train()
        self.discriminator.train()

        generator_loss = 0.0
        discriminator_real_loss = 0.0
        discriminator_fake_loss = 0.0
        fake_mean = 0.0
        fake_stdev = 0.0

        # First train the discriminator
        train_iterator = self.train_iterator(self.train_dataset)
        noise_iterator = self.noise_iterator(self.noise)

        for _ in range(self.batches_per_epoch):
            self.discriminator_optimizer.zero_grad()

            batch = next(train_iterator)
            noise = next(noise_iterator)

            batch = nn_util.move_to_device(batch, self._cuda_devices[0])
            noise = nn_util.move_to_device(noise, self._cuda_devices[0])

            # extract features
            features = self.feature_extractor(batch)

            # Real example, want discriminator to predict 1.
            ones = nn_util.move_to_device(torch.ones((self.batch_size, 1)), self._cuda_devices[0])
            real_error = self.discriminator(features, ones)["loss"]
            real_error.backward()

            # Fake example, want discriminator to predict 0.
            fake_data = self.generator(noise["array"])["output"]
            zeros = nn_util.move_to_device(torch.zeros((self.batch_size, 1)), self._cuda_devices[0])
            fake_error = self.discriminator(fake_data, zeros)["loss"]
            fake_error.backward()

            discriminator_real_loss += real_error.sum().item()
            discriminator_fake_loss += fake_error.sum().item()

            self.discriminator_optimizer.step()

        # Now train the generator
        for _ in range(self.batches_per_epoch):
            self.generator_optimizer.zero_grad()

            noise = next(noise_iterator)
            noise = nn_util.move_to_device(noise, self._cuda_devices[0])

            generated = self.generator(noise["array"], self.discriminator)
            fake_data = generated["output"]
            fake_error = generated["loss"]
            fake_error.backward()

            fake_mean += fake_data.mean()
            fake_stdev += fake_data.std()

            generator_loss += fake_error.sum().item()

            self.generator_optimizer.step()

        return {
                "generator_loss": generator_loss,
                "discriminator_fake_loss": discriminator_fake_loss,
                "discriminator_real_loss": discriminator_real_loss,
                "mean": fake_mean / self.batches_per_epoch,
                "stdev": fake_stdev / self.batches_per_epoch
        }

    def train(self) -> Dict[str, Any]:
        with tqdm.trange(self.num_epochs) as epochs:
            for n_epoch in epochs:
                metrics = self.train_one_epoch(n_epoch)
                description = (f'gl: {metrics["generator_loss"]:.3f} '
                               f'dfl: {metrics["discriminator_fake_loss"]:.3f} '
                               f'drl: {metrics["discriminator_real_loss"]:.3f} '
                               f'mean: {metrics["mean"]:.2f} '
                               f'std: {metrics["stdev"]:.2f} ')
                epochs.set_description(description)
        return metrics

    @classmethod
    def from_params(cls,   # type: ignore
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False) -> 'GanTrainer':

        config_file = params.pop('config_file')
        training_file = params.pop('training_file')
        dev_file = params.pop('dev_file')
        test_file = params.pop('test_file')
        cuda_device = params.pop_int("cuda_device")

        # Data reader
        reader = DatasetReader.from_params(params.pop('data_reader'))
        train_dataset = reader.read(cached_path(training_file))
        validation_dataset = reader.read(cached_path(dev_file))

        noise_reader = DatasetReader.from_params(params.pop("noise_reader"))
        noise = noise_reader.read("")

        # Vocabulary
        vocab = Vocabulary.from_instances(train_dataset,
                                          # min_count={'tokens': 2},
                                          only_include_pretrained_words=True,
                                          max_vocab_size=config_file.max_vocab_size,
                                          )
        logging.info('Vocab size: %s' % vocab.get_vocab_size())

        # Iterator
        train_iterator = DataIterator.from_params(params.pop("training_iterator"))
        train_iterator.index_with(vocab)
        noise_iterator = DataIterator.from_params(params.pop("noise_iterator"))

        # Model
        feature_extractor = Model.from_params(params.pop("feature_extractor"), vocab=vocab)
        generator = Model.from_params(params.pop("generator"))
        discriminator = Model.from_params(params.pop("discriminator"))

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

        # Optimizer
        generator_optimizer = Optimizer.from_params(
            [[n, p] for n, p in generator.named_parameters() if p.requires_grad],
            params.pop("generator_optimizer"))

        discriminator_optimizer = Optimizer.from_params(
            [[n, p] for n, p in chain(discriminator.named_parameters(), feature_extractor.named_parameters()) if p.requires_grad],
            params.pop("discriminator_optimizer"))

        # training_util.move_optimizer_to_cuda(generator_optimizer)
        # training_util.move_optimizer_to_cuda(discriminator_optimizer)

        num_epochs = params.pop_int("num_epochs")
        batches_per_epoch = params.pop_int("batches_per_epoch")
        batch_size = params.pop_int("batch_size")

        params.pop("trainer")

        params.assert_empty(__name__)

        return cls(serialization_dir,
                   train_dataset,
                   noise,
                   feature_extractor,
                   generator,
                   discriminator,
                   train_iterator,
                   noise_iterator,
                   generator_optimizer,
                   discriminator_optimizer,
                   batches_per_epoch,
                   num_epochs,
                   batch_size,
                   cuda_device)
