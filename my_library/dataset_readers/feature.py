from typing import Iterator, Dict
import torch
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from allennlp.data.fields import ArrayField


@DatasetReader.register("feature")
class FeatureReader(DatasetReader):
    def __init__(self) -> None:
        super().__init__(lazy=True)

    def _read(self, path: str) -> Iterator[Instance]:
        features = torch.load(path)
        train_features = features['train_features']
        train_labels = features['train_labels']

        for i in range(len(train_features)):
            yield self.text_to_instance(train_features[i], train_labels[i])

    def text_to_instance(self, f, l) -> Instance:  # type: ignore
        return Instance({
            "feature": ArrayField(f),
            "label": ArrayField(l)
        })
