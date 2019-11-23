from typing import Iterator, Dict
import torch
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from allennlp.data.fields import ArrayField


@DatasetReader.register("feature")
class FeatureReader(DatasetReader):
    def __init__(self, f_type: str) -> None:
        super().__init__(lazy=False)
        self.f_type = f_type

    def _read(self, path: str) -> Iterator[Instance]:
        features = torch.load(path)
        _features = features[self.f_type + '_features']
        _labels = features[self.f_type + '_labels']

        for i in range(len(_features)):
            yield self.text_to_instance(_features[i], _labels[i])

    def text_to_instance(self, f, l) -> Instance:  # type: ignore
        return Instance({
            "tokens": ArrayField(f),
            "label": ArrayField(l)
        })
