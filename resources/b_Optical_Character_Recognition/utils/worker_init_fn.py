import random
import signal

import numpy as np
import torch


def worker_init_fn(x: int) -> None:
    """
    Функция инициализации для класса DataLoader.

    Args:
        x: id воркера
    """
    seed = (int(torch.initial_seed()) + x) % (2 ** 32 - 1)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
