from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Item:
    """
    Датакласс для хранения одного сэмпла для детекции текста.
    Хранит путь к изображению, изображение, список полигонов с текстом, (опционально) список слов на изображении,
    (опционально) список полигонов с символами, (опционально) список символов, (опционально) список полигонов линий,
    (опционально) список полигонов групп.
    """
    fname: str
    img: np.ndarray
    word_polygons: List[np.ndarray]
    words: Optional[List[str]]
    char_polygons: Optional[List[np.ndarray]]
    chars: Optional[List[str]]
    line_polygons: Optional[List[np.ndarray]] = None
    group_polygons: Optional[List[np.ndarray]] = None
