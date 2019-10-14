import logging
import random
import os
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


def experiment_trec(_phase):
    """
    phase = 'cls_on_real'
    phase = 'gan'
    phase = 'cls_on_fake'
    """
    config_file = TRECConfig
    hparam = config_file.hparam
    model_real_dir = os.path.join(config_file.model_dir,
                             '_'.join(['ph', config_file.phase_real,
                                       'lr', str(hparam['lr']),
                                       'bs', str(hparam['batch_size']),
                                       'h', str(hparam['d_hidden']),
                                       'dp', str(hparam['dropout']),
                                       'frac', str(hparam['file_frac'])
                                       ]))
    model_gan_dir = os.path.join(config_file.model_dir,
                                  '_'.join(['ph', config_file.phase_gan,
                                            'lr', str(hparam['lr']),
                                            'bs', str(hparam['batch_size']),
                                            'h', str(hparam['d_hidden']),
                                            'dp', str(hparam['dropout']),
                                            'frac', str(hparam['file_frac'])
                                            ]))
    model_fake_dir = os.path.join(config_file.model_dir,
                                  '_'.join(['ph', config_file.phase_fake,
                                            'lr', str(hparam['lr']),
                                            'bs', str(hparam['batch_size']),
                                            'h', str(hparam['d_hidden']),
                                            'dp', str(hparam['dropout']),
                                            'frac', str(hparam['file_frac'])
                                            ]))
    if _phase == config_file.phase_real:
        model_dir = model_real_dir
    elif _phase == config_file.phase_gan:
        model_dir = model_gan_dir
    elif _phase == config_file.phase_fake:
        model_dir = model_fake_dir
    else:
        raise ValueError('unknown training phase name %s.' % _phase)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    n_classes = config_file.n_label
    batch_size = hparam['batch_size']
    lr = hparam['lr']
    d_hidden = hparam['d_hidden']
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
                    "lr": 0.00005
                },
                "discriminator_optimizer": {
                    "type": "rmsprop",
                    "lr": 0.00005
                },
                "classifier_optimizer": {
                    "type": "adam",
                    "lr": lr
                }
            },
            "batch_size": batch_size,
            "cuda_device": cuda_device,
            "patience": patience,

            "n_epoch_real": 1000,
            "n_epoch_gan": 100,
            "n_epoch_fake": 1000,

            "batch_per_epoch": batch_per_epoch,
            "num_loop_discriminator": 5,
            "num_loop_classifier_on_fake": 100,
            "clip_value": 1,
            "n_classes": config_file.n_label,
            "phase": _phase,
            "model_real_dir": model_real_dir,
            "model_gan_dir": model_gan_dir,
            "model_fake_dir": model_fake_dir
        })

    import tempfile
    # serialization_dir_ = tempfile.mkdtemp()
    trainer_ = TrainerBase.from_params(params_, model_dir)
    ###########
    # Training
    ###########
    train_metrics, meta_data_train = trainer_.train()
    pprint(train_metrics)
    # save training meta data
    print('[saving] training meta data ...')

    if _phase == 'cls_on_real':
        torch.save(meta_data_train, config_file.train_real_meta_path % (config_file.corpus_name, hparam['file_frac']))
    elif _phase == 'gan':
        torch.save(meta_data_train, config_file.train_gan_meta_path % (config_file.corpus_name, hparam['file_frac']))
    elif _phase == 'cls_on_fake':
        torch.save(meta_data_train, config_file.train_fake_meta_path % (config_file.corpus_name, hparam['file_frac']))
    else:
        raise ValueError('unknown training phase name %s.' % _phase)
    print('[saved]')

    #######
    # Test
    #######
    if _phase == 'cls_on_real' or _phase == 'cls_on_fake':
        test_metrics, meta_data_test = trainer_.test()
        pprint(test_metrics)
        print('[saving] test meta data ...')
        torch.save(meta_data_test, config_file.test_meta_path % (config_file.corpus_name, hparam['file_frac']))
        print('[saved]')


if __name__ == '__main__':
    # phase = 'cls_on_real'
    phase = 'gan'
    # phase = 'cls_on_fake'
    experiment_trec(phase)
