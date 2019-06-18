import logging
import numpy as np
import random
import os
import shutil
import torch
import torch.optim as optim

from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params
from allennlp.data.token_indexers import PretrainedBertIndexer, SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer, TrainerBase
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper

from my_library.models.data_augmentation import FeatureExtractor, Generator
from my_library.dataset_readers.stance import StanceDatasetReader
from config import StanceConfig

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

torch.manual_seed(2019)
random.seed(2019)


class TrainLSTM(object):

    def __init__(self, training_file, dev_file, test_file, model_path, hparam, reader, config_file):
        self.training_file = training_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.model_path = model_path
        self.hparam = hparam
        self.config_file = config_file

        self.reader = reader(
            lazy=True,
            tokenizer=lambda string: string.split()[:config_file.max_seq_len],
            token_indexers={"tokens": SingleIdTokenIndexer()}
        )
        logger.info('loading training data ...')
        self.train_dataset = self.reader.read(cached_path(self.training_file))

        logger.info('loading validation data ...')
        self.validation_dataset = self.reader.read(cached_path(self.dev_file))

        logger.info('building vocabulary ...')
        self.vocab = Vocabulary.from_instances(self.train_dataset,
                                               # min_count={'tokens': 2},
                                               only_include_pretrained_words=True,
                                               max_vocab_size=config_file.max_vocab_size,
                                               pretrained_files={'tokens': config_file.GLOVE_840B_300D},
                                               )
        logger.info('Vocab size: %s' % self.vocab.get_vocab_size())

        self.train_iterator = BucketIterator(batch_size=self.hparam['batch_size'],
                                             # instances_per_epoch=instances_per_epoch,
                                             # max_instances_in_memory=32000,
                                             sorting_keys=[('tokens', 'num_tokens')],
                                             track_epoch=True,
                                             )
        self.train_iterator.index_with(self.vocab)

        word_embedding_layer: TextFieldEmbedder = BasicTextFieldEmbedder(
            {"tokens": Embedding(num_embeddings=self.vocab.get_vocab_size('tokens'),
                                 embedding_dim=self.hparam['d_word_emb'])},
            allow_unmatched_keys=True)

        encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(
            input_size=self.hparam['d_word_emb'],
            hidden_size=self.hparam['d_rnn'],
            num_layers=2,
            dropout=self.hparam['dropout'],
            bidirectional=True, batch_first=True))

        self.feature_extractor = FeatureExtractor(
            word_embedding_layer=word_embedding_layer,
            encoder=encoder,
            d_hidden=self.hparam['d_hidden'],
            dropout=self.hparam['dropout'],
        ).cuda(self.hparam['cuda_device'])

        self.generator = Generator(
            d_hidden=self.hparam['d_hidden'],
            dropout=self.hparam['dropout'],
        )

    def train(self):
        if os.path.exists(self.config_file.model_path):
            shutil.rmtree(self.config_file.model_path)

        optimizer = optim.Adam(self.gan.parameters(), lr=self.hparam['lr'])
        # lr_scheduler = NoamLR(optimizer=optimizer,
        #                       model_size=self.hparam['d_hidden'],
        #                       warmup_steps=100)

        trainer = Trainer(
            model=self.gan,
            optimizer=optimizer,
            # learning_rate_scheduler=lr_scheduler,
            iterator=self.train_iterator,
            train_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            validation_metric='-loss',
            cuda_device=self.hparam['cuda_device'],
            num_epochs=self.hparam['epochs'],
            patience=self.hparam['patience'],
            num_serialized_models_to_keep=1,
            serialization_dir=self.model_path
        )

        trainer.train()

    def test(self):
        logger.info('Testing ...')
        logger.info('loading test set ...')
        test_dataset = self.reader.read(cached_path(self.test_file))
        y_true = np.array([self.vocab.get_token_index(sample.fields['labels'].label,
                                                      namespace='labels')
                           for sample in test_dataset])
        test_iterator = BasicIterator(batch_size=128, track_epoch=True)
        test_iterator.index_with(self.vocab)
        logger.info('loading best model ...')
        with open(os.path.join(self.model_path, 'best.th'), 'rb') as f:
            self.model.load_state_dict(torch.load(f))
        logger.info('predicting ...')
        predictor = Predictor(self.model, test_iterator, self.hparam['cuda_device'])
        test_pred_probs = predictor.predict(test_dataset)
        test_pred_labels = np.argmax(test_pred_probs, axis=1)
        print(metrics.classification_report(y_true, test_pred_labels))
        return {
            'micro': metrics.f1_score(y_true, test_pred_labels, average='micro'),
            'macro': metrics.f1_score(y_true, test_pred_labels, average='macro'),
            'macro2': metrics.f1_score(y_true, test_pred_labels, average='macro', labels=[0, 1]),
            'accuracy': metrics.accuracy_score(y_true, test_pred_labels)
        }


def experiment_stance():
    config_file = StanceConfig
    stance_target = 'a'
    hparam = config_file.hparam[stance_target]
    model_path = config_file.model_path % '_'.join(['tgt', stance_target,
                                                    'lambda', str(hparam['lambda']),
                                                    'lr', str(hparam['lr']),
                                                    'bs', str(hparam['batch_size']),
                                                    'h', str(hparam['d_hidden']),
                                                    'dropout', str(hparam['dropout']),
                                                    'frac', str(hparam['file_frac'])])

    params_ = Params(
        {
            "config_file": config_file,
            "stance_target": stance_target,
            "training_file": config_file.train_ratio_path % (stance_target, hparam['file_frac']),
            "dev_file": config_file.dev_ratio_path % (stance_target, hparam['file_frac']),
            "test_file": config_file.test_path % stance_target,
            "trainer": {
                "type": "gan-base"
            },

            # Readers
            "data_reader": {"type": "stance",
                            "lazy": True,
                            "token_indexers": {
                                "type": "single_id"
                            }},
            "noise_reader": {"type": "sampling", "sampler": {"type": "uniform"}},

            # Iterators
            "training_iterator": {
                "type": "bucket",
                "batch_size": config_file.hparam[stance_target]['batch_size'],
                "sorting_keys": [('tokens', 'num_tokens')],
                "track_epoch": False
            },
            "noise_iterator": {
                "type": "basic",
                "batch_size": config_file.hparam[stance_target]['batch_size']
            },

            # Modules
            "feature_extractor": {
                "type": "feature-extractor-base",
                "text_field_embedder": {
                    "type": "basic",
                    "token_embedders": {"tokens": {
                        "type": "embedding",
                    }},
                },
                "text_encoder": {
                    "type": "lstm",
                    "input_size": config_file.hparam[stance_target]['d_word_emb'],
                    "hidden_size": config_file.hparam[stance_target]['d_rnn'],
                    "num_layers": 2,
                    "batch_first": True,
                    "dropout": config_file.hparam[stance_target]['dropout'],
                    "bidirectional": True,
                },
                "d_hidden": 128,
                "dropout": 0.1
            },
            "generator": {
                "type": "generator-base",
                "input_dim": 1,
                "hidden_dim": 5,
                "output_dim": 1
            },
            "discriminator": {
                "type": "discriminator-base",
                "input_dim": 128,
                "hidden_dim": 10,
                "preprocessing": "moments"
            },
            "generator_optimizer": {"type": "sgd", "lr": 0.1},
            "discriminator_optimizer": {"type": "sgd", "lr": 0.1},
            "num_epochs": 1000,
            "batches_per_epoch": 2
        })

    import tempfile
    serialization_dir_ = tempfile.mkdtemp()
    trainer_ = TrainerBase.from_params(params_, serialization_dir_)
    metrics_ = trainer_.train()
    print(metrics_)


if __name__ == '__main__':
    # python -m allennlp.tests.training.gan_trainer_test
    #
    # pylint: disable=invalid-name
    experiment_stance()
