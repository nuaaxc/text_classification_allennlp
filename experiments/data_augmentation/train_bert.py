import logging
import random
import numpy as np
import torch
import my_library
from config.stance import *
from config.sst import *
from config.r8 import *
from config.offensive import *
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


def sst():
    if SSTCfg.hp.phase == DirCfg.phase_real_str:
        run_real(SSTCfg)
    elif SSTCfg.hp.phase == DirCfg.phase_gan_str:
        run_gan(SSTCfg)
    elif SSTCfg.hp.phase == DirCfg.phase_fake_str:
        run_fake(SSTCfg)
    else:
        raise ValueError('phase name not found.')


def r8():
    if R8Cfg.hp.phase == DirCfg.phase_real_str:
        run_real(R8Cfg)
    elif R8Cfg.hp.phase == DirCfg.phase_gan_str:
        run_gan(R8Cfg)
    elif R8Cfg.hp.phase == DirCfg.phase_fake_str:
        run_fake(R8Cfg)
    else:
        raise ValueError('phase name not found.')


def offensive():
    if OffensiveCfg.hp.phase == DirCfg.phase_real_str:
        run_real(OffensiveCfg)
    elif OffensiveCfg.hp.phase == DirCfg.phase_gan_str:
        run_gan(OffensiveCfg)
    elif OffensiveCfg.hp.phase == DirCfg.phase_fake_str:
        run_fake(OffensiveCfg)
    else:
        raise ValueError('phase name not found.')


if __name__ == '__main__':
    # stance()
    # sst()
    # r8()
    offensive()

