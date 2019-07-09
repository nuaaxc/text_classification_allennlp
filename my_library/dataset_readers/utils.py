from typing import List, Iterable, Any
import random
import numpy as np


from allennlp.common import Registrable
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import ArrayField, LabelField

random.seed(2019)


class InputSampler(Registrable):
    """
    Abstract base class for sampling from a distribution.
    """
    def sample(self, *dims: int) -> np.ndarray:
        raise NotImplementedError


@InputSampler.register('uniform')
class UniformSampler(InputSampler):
    """
    Sample from the uniform [0, 1] distribution.
    """
    def sample(self, *dims: int) -> np.ndarray:
        return np.random.uniform(0, 1, dims)


@InputSampler.register('normal')
class NormalSampler(InputSampler):
    """
    Sample from the normal distribution.
    """
    def __init__(self, mean: float = 0, stdev: float = 1.0) -> None:
        self.mean = mean
        self.stdev = stdev

    def sample(self, *dims: int) -> np.ndarray:
        return np.random.normal(self.mean, self.stdev, dims)


@DatasetReader.register("sampling")
class SamplingReader(DatasetReader):
    """
    A dataset reader that just samples from the provided sampler forever.
    """
    def __init__(self,
                 sampler: InputSampler,
                 label_set: List,
                 dim: int) -> None:
        super().__init__(lazy=True)
        self.sampler = sampler
        self.dim = dim
        self.label_set = label_set

    def _read(self, _: str) -> Iterable[Instance]:
        while True:
            example = self.sampler.sample(self.dim)
            yield self.text_to_instance(example)

    def text_to_instance(self, example: np.ndarray) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        return Instance({"array": ArrayField(example),
                         "label": LabelField(random.choice(self.label_set))})
