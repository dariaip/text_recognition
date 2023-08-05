import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


def unclip_polygon(polygon: np.ndarray, unclip_ratio: float) -> np.ndarray:
    """
    Auxiliary function for expanding the provided polygon in unclip_ratio times.

    Args:
        polygon: a polygon
        unclip_ratio: multiplicator for scaling

    Returns: a scaled polygon

    """
    if cv2.contourArea(polygon) == 0:
        return polygon

    poly = Polygon(polygon)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(polygon, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance)).astype(int)
    return expanded
