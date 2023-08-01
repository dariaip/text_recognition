from typing import List

import numpy as np

from .unclip_polygon import unclip_polygon


def unclip(polygons: List[np.ndarray], unclip_ratio: float) -> List[np.ndarray]:
    """
    Вспомогательная функция для расширения всех полигонов из входного списка polygons в unclip_ratio раз.
    Внутри вызывает функцию unclip_polygon для каждого полигона.

    Args:
        polygons: список полигонов
        unclip_ratio: значение для расширения

    Returns:
        список расширенных полигонов
    """
    return [unclip_polygon(p, unclip_ratio)[0] for p in polygons]
