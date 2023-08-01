import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Функция для сохранения случайных состояний во всех используемых библиотеках.

    Args:
        seed: случайное состояние
    """

    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("Seed is set to ", seed)
