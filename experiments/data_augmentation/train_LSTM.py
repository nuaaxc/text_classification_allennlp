import logging
import random
from pprint import pprint
import torch

from allennlp.common.params import Params
from allennlp.training.trainer import Trainer, TrainerBase

from my_library.models.data_augmentation import FeatureExtractor

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
    n_classes = config_file.n_label
    batch_size = config_file.hparam[stance_target]['batch_size']
    lr = config_file.hparam[stance_target]['lr']
    d_word_emb = config_file.hparam[stance_target]['d_word_emb']
    d_hidden = config_file.hparam[stance_target]['d_hidden']
    d_rnn = config_file.hparam[stance_target]['d_rnn']
    dropout = config_file.hparam[stance_target]['dropout']
    cuda_device = config_file.hparam[stance_target]['cuda_device']
    patience = config_file.hparam[stance_target]['patience']

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
            "vocab_reader": {"type": "stance",
                             "lazy": False,
                             "is_train": False,
                             },
            "train_reader": {"type": "stance",
                             "lazy": True,
                             "is_train": True,
                             },
            "val_reader": {"type": "stance",
                           "lazy": False,
                           "is_train": False,
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
                "sorting_keys": [('text', 'num_tokens')],
                "track_epoch": False
            },
            "noise_iterator": {
                "type": "basic",
                "batch_size": batch_size
            },

            # Modules
            "feature_extractor": {
                "type": "feature-extractor-base",
                "text_field_embedder": {
                    "tokens": {
                        "type": "embedding",
                        # "pretrained_file": config_file.GLOVE_TWITTER_27B_200D,
                        "pretrained_file": config_file.GLOVE_840B_300D,
                        "embedding_dim": d_word_emb,
                        "trainable": False
                    },
                },
                "text_encoder": {
                    "type": "lstm",
                    "input_size": d_word_emb,
                    "hidden_size": d_rnn,
                    "num_layers": 1,
                    "batch_first": True,
                    "dropout": dropout,
                    "bidirectional": True,
                },
                "feed_forward": {
                    "input_dim": 2 * d_rnn,
                    "num_layers": 2,
                    "hidden_dims": d_hidden,
                    "activations": "relu",
                    "dropout": dropout
                },
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
                    "num_layers": 2,
                    "hidden_dims": [d_hidden, 1],
                    "activations": "relu",
                    "dropout": dropout
                },
            },
            "classifier": {
                "type": "classifier-base",
                "feed_forward": {
                    "input_dim": d_hidden,
                    "num_layers": 2,
                    "hidden_dims": [d_hidden, 3],
                    "activations": "relu",
                    "dropout": dropout
                },
            },
            "optimizer": {
                "type": "gan",
                "generator_optimizer": {
                    "type": "adam",
                    "lr": lr
                },
                "discriminator_optimizer": {
                    "type": "adam",
                    "lr": lr
                },
                "classifier_optimizer": {
                    "type": "adam",
                    "lr": lr
                }
            },
            "num_epochs": 100,
            # "batches_per_epoch": 100,
            "batch_size": batch_size,
            "cuda_device": cuda_device,
            "patience": patience,
            "num_loop_discriminator": 10,
            "num_loop_generator": 10,
            "num_loop_classifier_on_real": 100,
            "num_loop_classifier_on_fake": 100,
        })

    import tempfile
    serialization_dir_ = tempfile.mkdtemp()
    trainer_ = TrainerBase.from_params(params_, serialization_dir_)
    train_metrics = trainer_.train()
    pprint(train_metrics)
    test_metrics = trainer_.test()
    pprint(test_metrics)


if __name__ == '__main__':
    # python -m allennlp.tests.training.gan_trainer_test
    #
    # pylint: disable=invalid-name
    experiment_stance()
