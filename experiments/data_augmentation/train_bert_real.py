import shutil
from pprint import pprint
import os
import torch

from allennlp.common.params import Params
from allennlp.training.trainer import TrainerBase


def run(cfg):
    model_dir = os.path.join(cfg.model_dir,
                             '_'.join(['ph', cfg.phase_real_str,
                                       'bs', str(cfg.hp.batch_size),
                                       'h', str(cfg.hp.d_hidden),
                                       'r', str(cfg.hp.file_ratio)
                                       ]))

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    batch_size = cfg.hp.batch_size
    lr = cfg.hp.lr
    cuda_device = cfg.hp.cuda_device
    patience = cfg.hp.patience
    max_pieces = cfg.hp.max_pieces

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
                "tokenizer": {
                    "word_splitter": "bert-basic"
                },
                "token_indexers": {
                    "bert": {
                        "type": "bert-pretrained",
                        "pretrained_model": cfg.BERT_VOC,
                        "max_pieces": max_pieces,
                    }
                }
            },
            "iterator": {
                "type": "bucket",
                "batch_size": batch_size,
                "sorting_keys": [('tokens', 'num_tokens')],
                "track_epoch": False
            },
            "model": {
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
                            "requires_grad": True
                        }
                    }
                },
                "seq2vec_encoder": {
                    "type": "bert_pooler",
                    "pretrained_model": cfg.BERT_MODEL,
                    "requires_grad": True
                },
                "feature_only": False
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
    features = trainer_.feature_collection()
    print(len(features['train_features']))
    print(len(features['train_labels']))
    print(len(features['validation_features']))
    print(len(features['validation_labels']))
    print(len(features['test_features']))
    print(len(features['test_labels']))
    print('[saving] features ...')
    torch.save(features, cfg.train_real_meta_path % (cfg.corpus_name, cfg.hp.file_ratio))
    print('[saved]')

