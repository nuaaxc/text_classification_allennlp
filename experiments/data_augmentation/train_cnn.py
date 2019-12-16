import logging
import random
import numpy as np
import torch

import my_library

from config.common import *
from config.stance import StanceCfgCNN
from config.sst import SSTCfgCNN
from config.r8 import *
from config.offensive import *
from config.trec import *

from experiments.data_augmentation.train_cnn_real import run as run_real

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

seed = 2020
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def stance():
    if StanceCfgCNN.hp.phase == DirCfg.phase_real_str:
        run_real(StanceCfgCNN)
    else:
        raise ValueError('phase name not found.')


def sst():
    if SSTCfgCNN.hp.phase == DirCfg.phase_real_str:
        run_real(SSTCfgCNN)
    else:
        raise ValueError('phase name not found.')


def r8():
    if R8Cfg.hp.phase == DirCfg.phase_real_str:
        run_real(R8Cfg)
    else:
        raise ValueError('phase name not found.')


def offensive():
    if OffensiveCfg.hp.phase == DirCfg.phase_real_str:
        run_real(OffensiveCfg)
    else:
        raise ValueError('phase name not found.')


def trec():
    if TRECCfg.hp.phase == DirCfg.phase_real_str:
        run_real(TRECCfg)
    else:
        raise ValueError('phase name not found.')


if __name__ == '__main__':
    stance()
    # sst()
    # r8()
    # offensive()
    # trec()

