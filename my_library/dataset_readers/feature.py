from typing import Iterator, Dict
import torch
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from allennlp.data.fields import ArrayField


@DatasetReader.register("feature")
class FeatureReader(DatasetReader):
    def __init__(self,
                 meta_path: str,
                 corpus_name: str,
                 file_frac: int) -> None:
        super().__init__(lazy=True)
        self.meta_path = meta_path
        self.corpus_name = corpus_name
        self.file_frac = file_frac

    def _read(self, _: str) -> Iterator[Instance]:
        features = torch.load(self.meta_path % (self.corpus_name, self.file_frac))['training_features']
        for f in features:
            yield self.text_to_instance(f)

    def text_to_instance(self, f) -> Instance:  # type: ignore
        return Instance({"feature": ArrayField(f['feature']),
                         "label": ArrayField(f['label'])})
