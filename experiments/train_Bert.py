import logging
import numpy as np
import random
import torch
import os
import shutil
import torch.optim as optim
from sklearn import metrics

from allennlp.common.file_utils import cached_path

from allennlp.data.token_indexers import PretrainedBertIndexer, SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.data.iterators import BasicIterator, BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.training.learning_rate_schedulers import NoamLR

from my_library.models import BertSentencePooling
from my_library.models import BaseLSTM
from my_library.predictors import Predictor
from my_library.dataset_readers.yahoo import YahooDatasetReader
from config import YahooConfig

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

torch.manual_seed(2019)
random.seed(2019)

lr = 0.0001
epochs = 50
patience = 5
batch_size = 256
d_hidden = 512
dropout = 0.1
# _lambda = 0.0
_lambda = 0.1
cuda_device = 0
config_file = YahooConfig


class TrainBert(object):

    def __init__(self):
        token_indexer = PretrainedBertIndexer(
            pretrained_model=config_file.BERT_VOC,
            # max_pieces=config_file.max_seq_len,
            do_lowercase=True,
        )
        self.reader = YahooDatasetReader(
            tokenizer=lambda string: token_indexer.wordpiece_tokenizer(string)[:config_file.max_seq_len],
            token_indexers={"tokens": token_indexer}
        )
        logger.info('loading training data ...')
        self.train_dataset = self.reader.read(cached_path(config_file.train_ratio_path % '100'))

        self.n_batch = int(len(list(self.train_dataset)) / batch_size)

        logger.info('loading validation data ...')
        self.validation_dataset = self.reader.read(cached_path(config_file.dev_ratio_path % '100'))

        logger.info('building vocabulary ...')
        self.vocab = Vocabulary.from_instances(self.train_dataset,
                                               # min_count={'tokens': 2},
                                               # only_include_pretrained_words=True,
                                               max_vocab_size=config_file.max_vocab_size
                                               )
        logger.info('Vocab size: %s' % self.vocab.get_vocab_size())

        self.train_iterator = BucketIterator(batch_size=batch_size,
                                             # biggest_batch_first=True,
                                             sorting_keys=[('tokens', 'num_tokens')],
                                             track_epoch=True,
                                             )
        self.train_iterator.index_with(self.vocab)

        word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder(
            {
                "tokens": PretrainedBertEmbedder(
                    pretrained_model=config_file.BERT_MODEL,
                    # requires_grad=True,
                    top_layer_only=True,  # conserve memory
                )},
            allow_unmatched_keys=True)

        encoder = BertSentencePooling(self.vocab, word_embeddings.get_output_dim())

        self.model = BaseLSTM(
            self.vocab,
            word_embeddings,
            encoder,
            d_hidden,
            dropout,
            cuda_device,
            _lambda
        ).cuda(cuda_device)

    def train(self):

        if os.path.exists(config_file.model_path):
            shutil.rmtree(config_file.model_path)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        lr_scheduler = NoamLR(optimizer=optimizer, model_size=768, warmup_steps=self.n_batch)

        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            learning_rate_scheduler=lr_scheduler,
            iterator=self.train_iterator,
            train_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            validation_metric='-loss',
            cuda_device=cuda_device,
            num_epochs=epochs,
            patience=patience,
            num_serialized_models_to_keep=2,
            serialization_dir=config_file.model_path
        )

        trainer.train()

    def test(self):
        logger.info('Testing ...')
        logger.info('loading test set ...')
        test_dataset = self.reader.read(cached_path(config_file.test_path))
        y_true = np.array([self.vocab.get_token_index(sample.fields['labels'].label,
                                                      namespace='labels')
                           for sample in test_dataset])
        test_iterator = BasicIterator(batch_size=128, track_epoch=True)
        test_iterator.index_with(self.vocab)
        logger.info('loading best model ...')
        with open(config_file.best_model_path, 'rb') as f:
            self.model.load_state_dict(torch.load(f))
        logger.info('predicting ...')
        predictor = Predictor(self.model, test_iterator, cuda_device)
        test_pred_probs = predictor.predict(test_dataset)
        test_pred_labels = np.argmax(test_pred_probs, axis=1)
        print(metrics.classification_report(y_true, test_pred_labels))
        return {
            'micro': metrics.f1_score(y_true, test_pred_labels, average='micro'),
            'macro': metrics.f1_score(y_true, test_pred_labels, average='macro'),
            'macro2': metrics.f1_score(y_true, test_pred_labels, average='macro', labels=[0, 1]),
            'accuracy': metrics.accuracy_score(y_true, test_pred_labels)
        }


if __name__ == '__main__':
    learning = TrainBert()
    learning.train()
    print(learning.test())
