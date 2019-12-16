import logging
import random
import numpy as np
import torch

import my_library

from config.common import *
from config.stance import *
from config.sst import SSTCfgLSTM
from config.r8 import *
from config.offensive import *
from config.trec import *

from experiments.data_augmentation.train_lstm_real import run as run_real

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

seed = 2020
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def stance():
    if StanceCfg.hp.phase == DirCfg.phase_real_str:
        run_real(StanceCfg)
    else:
        raise ValueError('phase name not found.')


def sst():
    if SSTCfgLSTM.hp.phase == DirCfg.phase_real_str:
        run_real(SSTCfgLSTM)
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
    # stance()
    sst()
    # r8()
    # offensive()
    # trec()

