from typing import Iterator, List, Dict
import logging
import random
from collections import defaultdict

import torch
import torch.optim as optim
import numpy as np

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField

from allennlp.data.dataset_readers import DatasetReader

from allennlp.common.file_utils import cached_path

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer

from config import DBPediaConfig

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def train_dev_split(sample_ratio=0.1, split_ratio=0.9):
    random.seed(2019)
    label_example = defaultdict(list)
    with open(DBPediaConfig.train_path, encoding='utf-8') as f:
        for line in f:
            # print(line)
            label, _, _ = line.strip().split(',"')
            label_example[label].append(line)

    training_size = None
    print('Training size for each class:')
    for label, example_list in label_example.items():
        training_size = len(example_list)
        print(label, training_size)

    sample_size = int(sample_ratio * training_size)
    print('Sample size:', sample_size)

    label_train_sample = defaultdict()
    label_dev_sample = defaultdict()
    for label, example_list in label_example.items():
        samples = random.sample(example_list, sample_size)
        random.shuffle(samples)
        label_train_sample[label] = samples[:int(len(samples) * split_ratio)]
        label_dev_sample[label] = samples[int(len(samples) * split_ratio):]

    print('Sampled training set:')
    for label, example_list in label_train_sample.items():
        print(label, len(example_list))
    print('Sampled dev set:')
    for label, example_list in label_dev_sample.items():
        print(label, len(example_list))

    # Check
    for label in label_example.keys():
        all_examples = label_example[label]
        sampled_train = label_train_sample[label]
        sampled_dev = label_dev_sample[label]
        assert len(set(sampled_train) & set(sampled_dev)) == 0
        assert len(set(sampled_train) & set(all_examples)) == len(set(sampled_train))
        assert len(set(sampled_dev) & set(all_examples)) == len(set(sampled_dev))

    # save to file
    print('saving sampled training set ...')
    with open(DBPediaConfig.train_ratio_path % int(sample_ratio*100), 'w', encoding='utf-8') as f:
        for label, samples in label_train_sample.items():
            f.writelines(samples)
    print('saving sampled dev set ...')
    with open(DBPediaConfig.dev_ratio_path % int(sample_ratio * 100), 'w', encoding='utf-8') as f:
        for label, samples in label_dev_sample.items():
            f.writelines(samples)
    print('saved.')


@DatasetReader.register("dbpedia")
class DBPediaDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                label, title, abstract = line.strip().split(',"')
                yield self.text_to_instance(title, abstract, label)

    def text_to_instance(self, title: str, abstract: str, label: str = None) -> Instance:  # type: ignore
        tokenized_title = self._tokenizer.tokenize(title)
        tokenized_abstract = self._tokenizer.tokenize(abstract)
        title_field = TextField(tokenized_title, self._token_indexers)
        abstract_field = TextField(tokenized_abstract, self._token_indexers)
        fields = {'title': title_field, 'abstract': abstract_field}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)


if __name__ == '__main__':
    train_dev_split(sample_ratio=0.1)
