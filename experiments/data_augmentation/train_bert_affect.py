import logging
import random
import os
import numpy as np
from pprint import pprint
import torch

from allennlp.common.params import Params
from allennlp.training.trainer import TrainerBase

import my_library

from config import AffectConfig, DirConfig

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logging.getLogger().setLevel(logging.INFO)

seed = 2020

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def experiment():
    cfg = AffectConfig
    _phase = cfg.hp.phase
    model_real_dir = os.path.join(cfg.model_dir,
                                  '_'.join(['ph', cfg.phase_real_str,
                                            'lr', str(cfg.hp.lr),
                                            'bs', str(cfg.hp.batch_size),
                                            'h', str(cfg.hp.d_hidden),
                                            'dp', str(cfg.hp.dropout),
                                            'r', str(cfg.hp.file_ratio)
                                            ]))
    model_gan_dir = os.path.join(cfg.model_dir,
                                 '_'.join(['ph', cfg.phase_gan_str,
                                           'lr', str(cfg.hp.lr),
                                           'bs', str(cfg.hp.batch_size),
                                           'h', str(cfg.hp.d_hidden),
                                           'dp', str(cfg.hp.dropout),
                                           'r', str(cfg.hp.file_ratio)
                                           ]))
    model_fake_dir = os.path.join(cfg.model_dir,
                                  '_'.join(['ph', cfg.phase_fake_str,
                                            'lr', str(cfg.hp.lr),
                                            'bs', str(cfg.hp.batch_size),
                                            'h', str(cfg.hp.d_hidden),
                                            'dp', str(cfg.hp.dropout),
                                            'r', str(cfg.hp.file_ratio)
                                            ]))
    if _phase == cfg.phase_real_str:
        model_dir = model_real_dir
    elif _phase == cfg.phase_gan_str:
        model_dir = model_gan_dir
    elif _phase == cfg.phase_fake_str:
        model_dir = model_fake_dir
    else:
        raise ValueError('unknown training phase name %s.' % _phase)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    n_classes = cfg.n_label
    batch_size = cfg.hp.batch_size
    lr = cfg.hp.lr
    d_hidden = cfg.hp.d_hidden
    dropout = cfg.hp.dropout
    cuda_device = cfg.hp.cuda_device
    patience = cfg.hp.patience
    conservative_rate = cfg.hp.conservative_rate
    batch_per_epoch = cfg.hp.batch_per_epoch
    batch_per_generator = cfg.hp.batch_per_generator
    gen_step = cfg.hp.gen_step
    n_epoch_gan = cfg.hp.n_epoch_gan

    params_ = Params(
        {
            "config_file": cfg,
            "training_file": cfg.train_ratio_path % cfg.hp.file_ratio,
            "dev_file": cfg.dev_path,
            "test_file": cfg.test_path,
            "trainer": {
                "type": "gan-bert"
            },

            # Readers
            "dataset_reader": {
                "lazy": False,
                "type": "text_dataset",
                "tokenizer": {
                    "word_splitter": "bert-basic"
                },
                "token_indexers": {
                    "tokens": {
                        "type": "bert-pretrained",
                        "pretrained_model": DirConfig.BERT_VOC,
                        "max_pieces": 128,
                    }
                }
            },
            "noise_reader": {
                "type": "sampling",
                "sampler": {"type": "normal"},
                "dim": d_hidden,
                "label_set": cfg.labels
            },
            "feature_reader": {
                "type": "feature",
                "meta_path": cfg.train_real_meta_path,
                "corpus_name": cfg.corpus_name,
                "file_frac": cfg.hp.file_ratio
            },
            # Iterators
            "training_iterator": {
                "type": "bucket",
                "batch_size": batch_size,
                "skip_smaller_batches": True,
                # "instances_per_epoch": batch_per_epoch * batch_size,
                # "max_instances_in_memory": batch_per_epoch * batch_size,
                "sorting_keys": [('text', 'num_tokens')],
                "track_epoch": False
            },
            "noise_iterator": {
                "type": "basic",
                "batch_size": batch_size
            },
            "feature_iterator": {
                "type": "basic",
                "batch_size": batch_size
            },
            # Modules
            "feature_extractor": {
                "type": "bert_encoder",
                "bert_path": DirConfig.BERT_MODEL,
                "dropout": dropout,
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
                "perturb": {
                    "input_dim": 2 * d_hidden,
                    "num_layers": 1,
                    "hidden_dims": d_hidden,
                    "activations": "relu",
                    "dropout": dropout
                },
                "feed_forward": {
                    "input_dim": 2 * d_hidden,
                    "num_layers": 1,
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
                    "num_layers": 3,
                    "hidden_dims": [d_hidden, d_hidden, n_classes],
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
            "conservative_rate": conservative_rate,

            "n_epoch_real": 1000,
            "n_epoch_gan": n_epoch_gan,
            "n_epoch_fake": 1000,

            "batch_per_epoch": batch_per_epoch,
            "batch_per_generator": batch_per_generator,
            "gen_step": gen_step,
            "num_loop_discriminator": 5,
            "clip_value": 1,
            "n_classes": cfg.n_label,

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
    meta_data_train = trainer_.train()
    pprint(meta_data_train['metrics'])
    # save training meta data
    print('[saving] training meta data ...')

    if _phase == 'cls_on_real':
        torch.save(meta_data_train, cfg.train_real_meta_path % (cfg.corpus_name, cfg.hp.file_ratio))
    elif _phase == 'gan':
        torch.save(meta_data_train, cfg.train_gan_meta_path % (cfg.corpus_name, cfg.hp.file_ratio))
    elif _phase == 'cls_on_fake':
        torch.save(meta_data_train, cfg.train_fake_meta_path % (cfg.corpus_name, cfg.hp.file_ratio))
    else:
        raise ValueError('unknown training phase name %s.' % _phase)
    print('[saved]')

    #######
    # Test
    #######
    if _phase == 'cls_on_real' or _phase == 'cls_on_fake':
        meta_data_test = trainer_.test()
        print('accuracy:', meta_data_test['accuracy'])
        print('micro:', meta_data_test['micro'])
        print('macro:', meta_data_test['macro'])
        print('[saving] test meta data ...')
        torch.save(meta_data_test, cfg.test_meta_path % (cfg.corpus_name, cfg.hp.file_ratio))
        print('[saved]')


if __name__ == '__main__':
    experiment()
