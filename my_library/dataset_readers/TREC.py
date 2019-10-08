from typing import Iterator, List, Dict
import logging

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField

from allennlp.data.dataset_readers import DatasetReader

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_filter import StopwordFilter


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("trec")
class TRECDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: WordTokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(word_filter=StopwordFilter())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                label, text = line.strip().split('\t')
                yield self.text_to_instance(label, text)

    def text_to_instance(self, label: str, text: str) -> Instance:  # type: ignore
        tokenized_text = self._tokenizer.tokenize(text)
        text_field = TextField(tokenized_text, self._token_indexers)
        fields = {'text': text_field}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)


if __name__ == '__main__':
    pass
