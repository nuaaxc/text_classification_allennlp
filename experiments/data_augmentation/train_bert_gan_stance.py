import logging
import random
import os
import shutil
import numpy as np
from pprint import pprint
import torch

from allennlp.common.params import Params
from allennlp.training.trainer import Trainer, TrainerBase

import my_library

from config import StanceConfig, DirConfig

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logging.getLogger().setLevel(logging.INFO)

seed = 2020

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def run():
    cfg = StanceConfig
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
    if os.path.exists(model_gan_dir):
        shutil.rmtree(model_gan_dir)

    if not os.path.exists(model_gan_dir):
        os.makedirs(model_gan_dir)

    n_classes = cfg.n_label
    batch_size = cfg.hp.batch_size
    d_hidden = cfg.hp.d_hidden
    dropout = cfg.hp.dropout
    cuda_device = cfg.hp.cuda_device
    patience = cfg.hp.patience
    conservative_rate = cfg.hp.conservative_rate
    batch_per_epoch = cfg.hp.batch_per_epoch
    batch_per_generator = cfg.hp.batch_per_generator
    gen_step = cfg.hp.gen_step
    n_epoch_gan = cfg.hp.n_epoch_gan
    best_cls_model_state_path = os.path.join(model_real_dir, 'best.th')

    params_ = Params(
        {
            "config_file": cfg,
            "train_feature_path": cfg.train_ratio_path % cfg.hp.file_ratio,

            "trainer": {
                "type": "gan"
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
            "noise_iterator": {
                "type": "basic",
                "batch_size": batch_size
            },
            "feature_iterator": {
                "type": "basic",
                "batch_size": batch_size
            },
            # Model
            "best_cls_model_state_path": best_cls_model_state_path,
            "cls_model": {
                "type": "feature_classifier",
                "text_field_embedder": {
                    "allow_unmatched_keys": True,
                    "embedder_to_indexer_map": {
                        "bert": ["bert", "bert-offsets"],
                    },
                    "token_embedders": {
                        "bert": {
                            "type": "bert-pretrained",
                            "pretrained_model": DirConfig.BERT_MODEL,
                            "top_layer_only": True,
                            "requires_grad": True
                        }
                    }
                },
                "seq2vec_encoder": {
                    "type": "bert_pooler",
                    "pretrained_model": DirConfig.BERT_MODEL,
                    "requires_grad": True
                }
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
            },
            "batch_size": batch_size,
            "cuda_device": cuda_device,
            "patience": patience,
            "conservative_rate": conservative_rate,
            "n_epoch_gan": n_epoch_gan,

            "batch_per_epoch": batch_per_epoch,
            "batch_per_generator": batch_per_generator,
            "gen_step": gen_step,
            "num_loop_discriminator": 5,
            "clip_value": 1,
        })
    exit()
    trainer_ = TrainerBase.from_params(params_, model_gan_dir)

    meta_data_train = trainer_.train()
    pprint(meta_data_train['metrics'])
    # save training meta data
    print('[saving] training meta data ...')
    torch.save(meta_data_train, cfg.train_gan_meta_path % (cfg.corpus_name, cfg.hp.file_ratio))
    print('[saved]')


if __name__ == '__main__':
    run()