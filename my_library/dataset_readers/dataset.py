from typing import Iterator, Dict
import random
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField, ArrayField

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_filter import StopwordFilter


@DatasetReader.register("text_dataset")
class TextDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: WordTokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 aug_num: int = None) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer or WordTokenizer(word_filter=StopwordFilter())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        self.aug_num = aug_num

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                if 'dev.txt' in file_path or 'test.txt' in file_path:
                    label, text = line.strip().split('\t')
                else:
                    _id, label, text = line.strip().split('\t')
                    if '.' in _id:
                        rnd = random.random()
                        if rnd > float(self.aug_num / 64):
                            continue
                yield self.text_to_instance(label, text)

    def text_to_instance(self, label: str, text: str) -> Instance:  # type: ignore
        tokenized_text = self._tokenizer.tokenize(text)
        text_field = TextField(tokenized_text, self._token_indexers)
        fields = {'tokens': text_field}
        if label is not None:
            fields['labels'] = LabelField(label)
        return Instance(fields)
