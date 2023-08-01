import cv2
import numpy as np


def fit_min_rot_box_on_polygon(polygon: np.ndarray) -> np.ndarray:
    """
    Вспомогательная функция для аппроксимации полигона повернутым прямоугольником (на основе метода cv2.minAreaRect).

    Args:
        polygon: полигон

    Returns:
        аппроксимация полигона повернутым прямоугольником
    """
    bounding_box = cv2.minAreaRect(polygon)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = np.array([points[index_1], points[index_2], points[index_3], points[index_4]])
    box = box.astype(int)
    return box
