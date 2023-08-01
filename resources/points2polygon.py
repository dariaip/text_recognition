from typing import Union, List

import numpy as np
from shapely.geometry import Polygon


def points2polygon(points: Union[List, np.ndarray], idx: int) -> Polygon:
    """
    Вспомогательный метод, превращающий набор точек в полигон типа shapely.geometry.Polygon.

    Args:
        points: набор точек
        idx: индекс, который присваивается полигону

    Returns:
        полигон типа shapely.geometry.Polygon
    """
    points = np.array(points).flatten()

    point_x = points[0::2]
    point_y = points[1::2]

    if len(point_x) < 4 or len(point_y) < 4:
        raise Exception('Implement approximation to 4 points as minimum')

    pol = Polygon(np.stack([point_x, point_y], axis=1)).buffer(0)
    #pol.idx = idx

    return pol
