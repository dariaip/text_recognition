from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Item:
    """
    Датакласс для хранения одного сэмпла для OCR.
    Хранит путь к изображению, текст на изображении, длину (длину текста или ширину изображения) и изображение.
    """
    image_fpath: str
    label: str
    length: int
    image: Optional[np.ndarray] = None
