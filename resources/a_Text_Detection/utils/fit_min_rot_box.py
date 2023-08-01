from typing import List

import numpy as np

from .fit_min_rot_box_on_polygon import fit_min_rot_box_on_polygon


def fit_min_rot_box(polygons: List[np.ndarray]) -> List[np.ndarray]:
    """
    Вспомогательная функция для аппроксимации всех полигонов из входного списка polygons повернутыми прямоугольниками.
    Внутри вызывает функцию fit_min_rot_box_on_polygon для каждого полигона.

    Args:
        polygons: список полигонов

    Returns:
        список аппроксимаций полигонов повернутыми прямоугольниками
    """
    return [fit_min_rot_box_on_polygon(p) for p in polygons]
