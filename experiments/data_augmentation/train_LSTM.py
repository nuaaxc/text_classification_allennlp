import logging
import numpy as np
import random
import os
import shutil

import torch
import torch.optim as optim
import torch.nn.functional as F

from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params
from allennlp.data.token_indexers import PretrainedBertIndexer, SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer, TrainerBase
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper

from my_library.models import FeatureExtractor, Generator
from my_library.dataset_readers.stance import StanceDatasetReader
from config import StanceConfig

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

torch.manual_seed(2019)
random.seed(2019)


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
            "training_file": config_file.train_ratio_path % (stance_target, hparam['file_frac']),
            "dev_file": config_file.dev_ratio_path % (stance_target, hparam['file_frac']),
            "test_file": config_file.test_path % stance_target,
            "trainer": {
                "type": "gan-base"
            },

            # Readers
            "data_reader": {"type": "stance",
                            "lazy": True
                            },
            "noise_reader": {
                "type": "sampling",
                "sampler": {"type": "uniform"},
                "dim": config_file.hparam[stance_target]['d_hidden']
            },

            # Iterators
            "training_iterator": {
                "type": "bucket",
                "batch_size": config_file.hparam[stance_target]['batch_size'],
                "sorting_keys": [('text', 'num_tokens')],
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
                    "tokens": {
                        "type": "embedding",
                        # "pretrained_file": config_file.GLOVE_TWITTER_27B_200D,
                        "pretrained_file": config_file.GLOVE_840B_300D,
                        "embedding_dim": config_file.hparam[stance_target]['d_word_emb'],
                        "trainable": False
                    },
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
                "feed_forward": {
                    "input_dim": 2 * config_file.hparam[stance_target]['d_rnn'],
                    "num_layers": 2,
                    "hidden_dims": config_file.hparam[stance_target]['d_hidden'],
                    "activations": "relu",
                    "dropout": config_file.hparam[stance_target]['dropout']
                },
            },
            "generator": {
                "type": "generator-base",
                "feed_forward": {
                    "input_dim": config_file.hparam[stance_target]['d_hidden'],
                    "num_layers": 2,
                    "hidden_dims": config_file.hparam[stance_target]['d_hidden'],
                    "activations": "relu",
                    "dropout": config_file.hparam[stance_target]['dropout']
                },
            },
            "discriminator": {
                "type": "discriminator-base",
                "d_input": config_file.hparam[stance_target]['d_hidden'],
                "d_hidden": config_file.hparam[stance_target]['d_hidden'],
                "n_layers": 2,
                "dropout": config_file.hparam[stance_target]['dropout'],
                'activation': 'relu',
                "preprocessing": None,
            },
            "generator_optimizer": {"type": "adam", "lr": 0.001},
            "discriminator_optimizer": {"type": "adam", "lr": 0.001},
            "num_epochs": 100,
            "batches_per_epoch": 100,
            "batch_size": config_file.hparam[stance_target]['batch_size'],
            "cuda_device": config_file.hparam[stance_target]['cuda_device'],
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
