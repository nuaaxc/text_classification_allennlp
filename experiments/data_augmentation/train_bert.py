import logging
import random
import numpy as np
from pprint import pprint
import torch

from allennlp.common.params import Params
from allennlp.training.trainer import Trainer, TrainerBase

import my_library

from config import TRECConfig, DirConfig

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logging.getLogger().setLevel(logging.INFO)

torch.manual_seed(2019)
random.seed(2019)
np.random.seed(2019)


def experiment_trec():
    config_file = TRECConfig
    hparam = config_file.hparam
    model_path = config_file.model_path % '_'.join(['lambda', str(hparam['lambda']),
                                                    'lr', str(hparam['lr']),
                                                    'bs', str(hparam['batch_size']),
                                                    'h', str(hparam['d_hidden']),
                                                    'dropout', str(hparam['dropout']),
                                                    'frac', str(hparam['file_frac'])])
    n_epochs = hparam['epochs']
    n_classes = config_file.n_label
    batch_size = hparam['batch_size']
    lr = hparam['lr']
    d_word_emb = hparam['d_word_emb']
    d_hidden = hparam['d_hidden']
    d_rnn = hparam['d_rnn']
    dropout = hparam['dropout']
    cuda_device = hparam['cuda_device']
    patience = hparam['patience']
    batch_per_epoch = hparam['batch_per_epoch']

    params_ = Params(
        {
            "config_file": config_file,
            "training_file": config_file.train_ratio_path % hparam['file_frac'],
            "dev_file": config_file.dev_ratio_path % hparam['file_frac'],
            "test_file": config_file.test_norm_path,
            "trainer": {
                "type": "gan-bert"
            },

            # Readers
            "dataset_reader": {
                "lazy": False,
                "type": "trec",
                "tokenizer": {
                    "word_splitter": "bert-basic"
                },
                "token_indexers": {
                    "tokens": {
                        "type": "bert-pretrained",
                        "pretrained_model": DirConfig.BERT_VOC
                    }
                }
            },
            "noise_reader": {
                "type": "sampling",
                "sampler": {"type": "uniform"},
                "dim": d_hidden,
                "label_set": config_file.labels
            },

            # Iterators
            "training_iterator": {
                "type": "bucket",
                "batch_size": batch_size,
                "instances_per_epoch": batch_per_epoch * batch_size,
                "sorting_keys": [('text', 'num_tokens')],
                "track_epoch": False
            },
            "noise_iterator": {
                "type": "basic",
                "batch_size": batch_size
            },

            # Modules
            "feature_extractor": {
                "type": "bert_encoder",
                "bert_path": DirConfig.BERT_MODEL,
                "dropout": dropout,
                # "trainable": False
                "trainable": True
            },
            "generator": {
                "type": "generator-base",
                "label_emb": {
                    "type": "embedding",
                    "num_embeddings": n_classes,
                    "embedding_dim": d_hidden,
                    "trainable": True
                },
                "feed_forward": {
                    "input_dim": 2 * d_hidden,
                    "num_layers": 2,
                    "hidden_dims": d_hidden,
                    "activations": "relu",
                    "dropout": dropout
                },
            },
            "discriminator": {
                "type": "discriminator-base",
                "label_emb": {
                    "type": "embedding",
                    "num_embeddings": n_classes,
                    "embedding_dim": d_hidden,
                    "trainable": True
                },
                "feed_forward": {
                    "input_dim": 2 * d_hidden,
                    "num_layers": 3,
                    "hidden_dims": [d_hidden, d_hidden, 1],
                    "activations": "relu",
                    "dropout": dropout
                },
            },
            "classifier": {
                "type": "classifier-base",
                "feed_forward": {
                    "input_dim": d_hidden,
                    "num_layers": 2,
                    "hidden_dims": [d_hidden, n_classes],
                    "activations": "relu",
                    "dropout": dropout
                },
            },
            "optimizer": {
                "type": "gan",
                "generator_optimizer": {
                    "type": "rmsprop",
                    "lr": lr
                },
                "discriminator_optimizer": {
                    "type": "rmsprop",
                    "lr": lr
                },
                "classifier_optimizer": {
                    "type": "adam",
                    "lr": lr
                }
            },
            "num_epochs": n_epochs,
            "batch_size": batch_size,
            "cuda_device": cuda_device,
            "patience": patience,
            "num_loop_discriminator": 20,
            "num_loop_generator": 4,
            "num_loop_classifier_on_real": batch_per_epoch,
            "num_loop_classifier_on_fake": 40,
            "clip_value": 1,
            # 'no_gen': True,
            'no_gen': False,
        })

    import tempfile
    serialization_dir_ = tempfile.mkdtemp()
    trainer_ = TrainerBase.from_params(params_, serialization_dir_)

    train_metrics, meta_data_train = trainer_.train()
    pprint(train_metrics)
    test_metrics, meta_data_test = trainer_.test()
    pprint(test_metrics)

    # save training meta data
    print('[saving] training meta data ...')
    torch.save(meta_data_train,
               config_file.train_meta_path % (config_file.corpus_name,
                                              hparam['file_frac']))
    print('saved.')

    print('[saving] test meta data ...')
    torch.save(meta_data_test,
               config_file.test_meta_path % (config_file.corpus_name,
                                             hparam['file_frac']))
    print('saved.')


if __name__ == '__main__':
    # python -m allennlp.tests.training.gan_trainer_test
    #
    # pylint: disable=invalid-name
    experiment_trec()
