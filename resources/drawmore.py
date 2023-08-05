from typing import List, Tuple

import cv2
import numpy as np


def draw_contours(
    image: np.ndarray,
    polygons: List[np.ndarray],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    fill: bool = False,
    inplace: bool = False
) -> np.ndarray:
    """
    Drawing a set of contours (polygons) on an image.

    Args:
        image: an image
        polygons: list of polygons
        color: color to use for the polygons' edges
        thickness: thickness of line around a polygon
        fill: whether to fill the area inside a polygon by the chosen color
        inplace: whether to change the initial image

    Returns:
        the initial image (if inplace=True), a copy of the initial image with added polygons otherwise
    """
    if not inplace:
        image = image.copy()

    for p in polygons:
        p = np.array([p]).astype(np.int32)
        if fill:
            cv2.fillPoly(image, p, color)
        else:
            cv2.drawContours(image, p, -1, color, thickness=thickness)
    return image
