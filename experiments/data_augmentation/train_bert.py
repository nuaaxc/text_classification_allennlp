import logging
import random
import numpy as np
import torch
import my_library
from config.stance import *
from config.common import *
from experiments.data_augmentation.train_bert_real import run as run_real
from experiments.data_augmentation.train_bert_gan import run as run_gan
from experiments.data_augmentation.train_bert_fake import run as run_fake

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

seed = 2020
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def stance():
    if StanceCfg.hp.phase == DirCfg.phase_real_str:
        run_real(StanceCfg)
    elif StanceCfg.hp.phase == DirCfg.phase_gan_str:
        run_gan(StanceCfg)
    elif StanceCfg.hp.phase == DirCfg.phase_fake_str:
        run_fake(StanceCfg)
    else:
        raise ValueError('phase name not found.')


if __name__ == '__main__':
    stance()
