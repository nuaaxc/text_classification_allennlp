from typing import Iterator, Dict
import logging
import random


from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField, MetadataField

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("stance")
class StanceDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 is_train: bool = True) -> None:
        super().__init__(lazy=lazy)
        self.tokenizer = tokenizer or WordTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._is_train = is_train

    def _read_train(self, file_path: str) -> Iterator[Instance]:
        F = []
        A = []
        N = []
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                label, text = line.strip().split('\t')
                if label == 'FAVOR':
                    F.append((label, text))
                elif label == 'AGAINST':
                    A.append((label, text))
                else:
                    N.append((label, text))

        while True:
            choice: float = random.random()
            if choice < 0.333:
                label, text = random.choice(F)
            elif choice >= 0.666:
                label, text = random.choice(A)
            else:
                label, text = random.choice(N)
            yield self.text_to_instance(text, label)

    def _read_val(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                label, text = line.strip().split('\t')
                yield self.text_to_instance(text, label)

    def _read(self, file_path: str) -> Iterator[Instance]:
        if self._is_train:
            return self._read_train(file_path)
        else:
            return self._read_val(file_path)

    def text_to_instance(self, text: str, label: str = None) -> Instance:  # type: ignore
        fields = {'text': TextField(self.tokenizer.tokenize(text), self.token_indexers)}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)


def baby_mention_stats():
    filename = 'C:/Users/nuaax/OneDrive/data61/project/stance_classification/dataset/semeval/data/la-all.txt'
    labels = []
    for line in open(filename, encoding='utf-8'):
        text = line.strip().split('\t')[2]
        label = line.strip().split('\t')[3]
        text = text.lower()
        if 'baby' in text:
            labels.append(label)
    import collections
    c = collections.Counter(labels)
    print(c)


if __name__ == '__main__':
    baby_mention_stats()





