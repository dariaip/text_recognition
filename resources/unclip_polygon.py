import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


def unclip_polygon(polygon: np.ndarray, unclip_ratio: float) -> np.ndarray:
    """
    Вспомогательная функция для расширения полигона в unclip_ratio раз.

    Args:
        polygon: полигон
        unclip_ratio: значение для расширения

    Returns: расширенный полигон

    """
    if cv2.contourArea(polygon) == 0:
        return polygon

    poly = Polygon(polygon)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(polygon, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance)).astype(int)
    return expanded
