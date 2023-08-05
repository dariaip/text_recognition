from typing import Union, List

import numpy as np
from shapely.geometry import Polygon


def points2polygon(points: Union[List, np.ndarray]) -> Polygon:
    """
    Auxiliary method that transforms a set of points into a polygon of type "shapely.geometry.Polygon".

    Args:
        points: a set of points

    Returns:
        shapely.geometry.Polygon object
    """
    points = np.array(points).flatten()

    point_x = points[0::2]
    point_y = points[1::2]

    if len(point_x) < 4 or len(point_y) < 4:
        raise Exception('Implement approximation to 4 points as minimum')

    pol = Polygon(np.stack([point_x, point_y], axis=1)).buffer(0)

    return pol
