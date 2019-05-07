from typing import Iterator, List, Dict, Callable
import logging


from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField, MetadataField

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("stance")
class StanceDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=lazy)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                label, text = line.strip().split('\t')
                yield self.text_to_instance([Token(x) for x in self.tokenizer(text)], label)

    def text_to_instance(self, tokens: List[Token], label: str = None) -> Instance:  # type: ignore
        fields = {'tokens': TextField(tokens, self.token_indexers)}
        if label is not None:
            fields['labels'] = LabelField(label)
        return Instance(fields)



