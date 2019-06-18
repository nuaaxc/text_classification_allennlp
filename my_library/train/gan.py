from typing import Dict, Iterable, Any
import logging
import tqdm
import torch
import numpy as np

from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params
from allennlp.data import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models import Model
from allennlp.training.trainer_base import TrainerBase
from allennlp.training.optimizers import Optimizer

from my_library.models.data_augmentation.feature_extractor import FeatureExtractor


@TrainerBase.register("gan-base")
class GanTrainer(TrainerBase):
    def __init__(self,
                 serialization_dir: str,
                 data: Iterable[Instance],
                 noise: Iterable[Instance],
                 generator: Model,
                 discriminator: Model,
                 iterator: DataIterator,
                 noise_iterator: DataIterator,
                 generator_optimizer: torch.optim.Optimizer,
                 discriminator_optimizer: torch.optim.Optimizer,
                 batches_per_epoch: int,
                 num_epochs: int) -> None:
        super(GanTrainer).__init__(serialization_dir, -1)
        self.data = data
        self.noise = noise
        self.generator = generator
        self.generator_optimizer = generator_optimizer
        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.num_epochs = num_epochs
        self.iterator = iterator
        self.noise_iterator = noise_iterator
        self.batches_per_epoch = batches_per_epoch

    def train_one_epoch(self) -> Dict[str, float]:
        self.generator.train()
        self.discriminator.train()

        generator_loss = 0.0
        discriminator_real_loss = 0.0
        discriminator_fake_loss = 0.0
        fake_mean = 0.0
        fake_stdev = 0.0

        # First train the discriminator
        data_iterator = self.iterator(self.data)
        noise_iterator = self.noise_iterator(self.noise)

        for _ in range(self.batches_per_epoch):
            self.discriminator_optimizer.zero_grad()

            batch = next(data_iterator)
            noise = next(noise_iterator)

            # Real example, want discriminator to predict 1.
            real_error = self.discriminator(batch["array"], torch.ones(1))["loss"]
            real_error.backward()

            # Fake example, want discriminator to predict 0.
            fake_data = self.generator(noise["array"])["output"]
            fake_error = self.discriminator(fake_data, torch.zeros(1))["loss"]
            fake_error.backward()

            discriminator_real_loss += real_error.sum().item()
            discriminator_fake_loss += fake_error.sum().item()

            self.discriminator_optimizer.step()

        # Now train the generator
        for _ in range(self.batches_per_epoch):
            self.generator_optimizer.zero_grad()

            noise = next(noise_iterator)
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
            for _ in epochs:
                metrics = self.train_one_epoch()
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
        stance_target = params.pop('stance_target')
        training_file = params.pop('training_file')
        dev_file = params.pop('dev_file')
        test_file = params.pop('test_file')

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
                                          pretrained_files={'tokens': config_file.GLOVE_TWITTER_27B_200D},
                                          )
        print('Vocab size: %s' % vocab.get_vocab_size())

        # Iterator
        training_iterator = DataIterator.from_params(params.pop("training_iterator"))
        training_iterator.index_with(vocab)

        noise_iterator = DataIterator.from_params(params.pop("noise_iterator"))

        # Feature_extractor
        feature_extractor = FeatureExtractor.from_params(params.pop("feature_extractor"))

        exit()
        # Generator
        generator = Model.from_params(params.pop("generator"))
        discriminator = Model.from_params(params.pop("discriminator"))



        generator_optimizer = Optimizer.from_params(
            [[n, p] for n, p in generator.named_parameters() if p.requires_grad],
            params.pop("generator_optimizer"))

        discriminator_optimizer = Optimizer.from_params(
            [[n, p] for n, p in discriminator.named_parameters() if p.requires_grad],
            params.pop("discriminator_optimizer"))

        num_epochs = params.pop_int("num_epochs")
        batches_per_epoch = params.pop_int("batches_per_epoch")
        params.pop("trainer")

        params.assert_empty(__name__)

        return cls(serialization_dir,
                   data,
                   noise,
                   generator,
                   discriminator,
                   iterator,
                   noise_iterator,
                   generator_optimizer,
                   discriminator_optimizer,
                   batches_per_epoch,
                   num_epochs)
