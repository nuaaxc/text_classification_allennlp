import shutil
from pprint import pprint
import os
import torch

from allennlp.common.params import Params
from allennlp.training.trainer import TrainerBase


def run(cfg):
    model_dir = os.path.join(cfg.model_dir,
                             '_'.join(['ph', cfg.phase_real_str,
                                       'r', str(cfg.hp.file_ratio),
                                       'base', cfg.model_name,
                                       ]))

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    batch_size = cfg.hp.batch_size
    lr = cfg.hp.lr
    cuda_device = cfg.hp.cuda_device
    patience = cfg.hp.patience
    d_rnn = cfg.hp.d_rnn
    d_dense = cfg.hp.d_dense
    dropout = cfg.hp.dropout
    n_label = cfg.n_label

    params_ = Params(
        {
            "config_file": cfg,
            "train_data_path": cfg.train_ratio_path % cfg.hp.file_ratio,
            "validation_data_path": cfg.dev_path,
            "test_data_path": cfg.test_path,

            # Readers
            "dataset_reader": {
                "lazy": False,
                "type": "text_dataset",
            },
            "iterator": {
                "type": "bucket",
                "batch_size": batch_size,
                "sorting_keys": [('tokens', 'num_tokens')],
                "track_epoch": False
            },
            "model": {
                "type": "text-classifier-base",
                "text_field_embedder": {
                    "tokens": {
                        "type": "embedding",
                        # "pretrained_file": config_file.GLOVE_TWITTER_27B_200D,
                        "pretrained_file": cfg.GLOVE_840B_300D,
                        "embedding_dim": cfg.GLOVE_840B_300D_DIM,
                        "trainable": False
                    }
                },
                "encoder": {
                    "type": "lstm",
                    "input_size": cfg.GLOVE_840B_300D_DIM,
                    "hidden_size": d_rnn,
                    "num_layers": 2,
                    "batch_first": True,
                    "bidirectional": True,
                    "dropout": dropout
                },
                "feed_forward": {
                    "input_dim": 2 * d_rnn,
                    "num_layers": 1,
                    "hidden_dims": n_label,
                    "activations": "relu",
                },
                "dropout": dropout,
            },
            "trainer": {
                "type": "cls-real",
                "optimizer": {
                    "type": "adam",
                    "lr": lr
                },
                "validation_metric": "-loss",
                "num_serialized_models_to_keep": 1,
                "num_epochs": 1000,
                # "grad_norm": 10.0,
                "patience": patience,
                "cuda_device": cuda_device
            },
        })

    trainer_ = TrainerBase.from_params(params_, model_dir)
    trainer_.train()
    res_test = trainer_.test()
    pprint(res_test)
    # features = trainer_.feature_collection()
    # print(len(features['train_features']))
    # print(len(features['train_labels']))
    # print(len(features['validation_features']))
    # print(len(features['validation_labels']))
    # print(len(features['test_features']))
    # print(len(features['test_labels']))
    # print('[saving] features ...')
    # torch.save(features, cfg.train_real_meta_path % (cfg.corpus_name, cfg.hp.file_ratio))
    # print('[saved]')
