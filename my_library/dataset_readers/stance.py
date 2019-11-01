from typing import Iterator, Dict
import logging
import torch

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField, ArrayField

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_filter import StopwordFilter

from config import StanceConfig

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("stance_dataset")
class StanceDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: WordTokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=lazy)
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


@DatasetReader.register("stance_feature")
class StanceFeatureReader(DatasetReader):
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


def merge_all_stance_target(input_file_path, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        f_out.write('ID\tTarget\tTweet\tStance\n')
        for stance_target in StanceConfig.target:
            print('processing %s ...' % stance_target)
            with open(input_file_path % stance_target, encoding='windows-1251') as f_in:
                next(f_in)
                for line in f_in:
                    f_out.write(line)


if __name__ == '__main__':
    # baby_mention_stats()
    # merge_all_stance_target(StanceConfig.train_raw_path, StanceConfig.train_raw_path_all)
    # merge_all_stance_target(StanceConfig.test_raw_path, StanceConfig.test_raw_path_all)
    pass



