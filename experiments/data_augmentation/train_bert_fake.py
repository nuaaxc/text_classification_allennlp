import shutil
import os
from pprint import pprint
import torch

from allennlp.common.params import Params
from allennlp.training.trainer import TrainerBase


def run(cfg):
    model_real_dir = os.path.join(cfg.model_dir,
                                  '_'.join(['ph', cfg.phase_real_str,
                                            'r', str(cfg.hp.file_ratio)
                                            ]))
    model_fake_dir = os.path.join(cfg.model_dir,
                                  '_'.join(['ph', cfg.phase_fake_str,
                                            'r', str(cfg.hp.file_ratio)
                                            ]))
    model_gan_dir = os.path.join(cfg.model_dir,
                                 '_'.join(['ph', cfg.phase_gan_str,
                                           'r', str(cfg.hp.file_ratio)
                                           ]))
    if os.path.exists(model_fake_dir):
        shutil.rmtree(model_fake_dir)

    if not os.path.exists(model_fake_dir):
        os.makedirs(model_fake_dir)

    n_classes = cfg.n_label
    batch_size = cfg.hp.batch_size
    d_hidden = cfg.hp.d_hidden
    dropout = cfg.hp.dropout
    lr = cfg.hp.lr
    cuda_device = cfg.hp.cuda_device
    patience = cfg.hp.patience
    max_pieces = cfg.hp.max_pieces
    gen_step = cfg.hp.gen_step
    best_gan_model_state_path = os.path.join(model_gan_dir, 'best.th')

    params_ = Params(
        {
            "config_file": cfg,
            "feature_path": cfg.train_real_meta_path % (cfg.corpus_name, cfg.hp.file_ratio),

            # Readers
            "noise_reader": {
                "type": "sampling",
                "sampler": {"type": "normal"},
                "dim": d_hidden,
                "label_set": cfg.labels
            },
            "train_feature_reader": {
                "type": "feature",
                "f_type": "train"
            },
            "validation_feature_reader": {
                "type": "feature",
                "f_type": "validation"
            },
            "test_feature_reader": {
                "type": "feature",
                "f_type": "test"
            },
            # Iterator
            "noise_iterator": {
                "type": "basic",
                "batch_size": batch_size
            },
            "feature_iterator": {
                "type": "bucket",
                "batch_size": batch_size,
                "sorting_keys": [('tokens', 'dimension_0')],
                "skip_smaller_batches": True
            },
            "vocab_path": os.path.join(model_real_dir, 'vocabulary'),

            # Model
            "cls": {
                "type": "feature_classifier",
                "text_field_embedder": {
                    "allow_unmatched_keys": True,
                    "embedder_to_indexer_map": {
                        "bert": ["bert", "bert-offsets"],
                    },
                    "token_embedders": {
                        "bert": {
                            "type": "bert-pretrained",
                            "pretrained_model": cfg.BERT_MODEL,
                            "top_layer_only": True,
                            "requires_grad": False
                        }
                    }
                },
                "seq2vec_encoder": {
                    "type": "bert_pooler",
                    "pretrained_model": cfg.BERT_MODEL,
                    "requires_grad": True
                },
                "num_labels": n_classes,
                "feature_only": True
            },
            "best_gan_model_state_path": best_gan_model_state_path,
            "gan": {
                "type": "gan-base",
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
            },
            "trainer": {
                "type": "cls-fake",
                "optimizer": {
                    "type": "adam",
                    "lr": lr
                },
                "validation_metric": "-loss",
                "num_serialized_models_to_keep": 1,
                "num_epochs": 1000,
                "patience": patience,
                "cuda_device": cuda_device,
            },
            "gen_step": gen_step,
        })

    trainer_ = TrainerBase.from_params(params_, model_fake_dir)
    _, gen_data = trainer_.train()
    res_test = trainer_.test()
    pprint(res_test)
    print('[saving] features ...')
    torch.save(gen_data, cfg.train_fake_meta_path % (cfg.corpus_name, cfg.hp.file_ratio))
    print('[saved]')
