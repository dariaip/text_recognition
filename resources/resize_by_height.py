from typing import Optional

import cv2
import numpy as np


def resize_by_height(image: np.ndarray, target_height: int, max_width: Optional[int] = None) -> np.ndarray:
    """
    Auxiliary method for scaling an image's height
    Final size of the image is defined based on target_height, whereas its width should be less (or equal) than max_width

    Args:
        image: an image
        target_height: height for scaling to
        max_width: (optional) maximal possible width of the scaled image

    Returns:
        the scaled image
    """
    h, w, _ = image.shape
    interpolation = cv2.INTER_AREA if h > target_height else cv2.INTER_LINEAR
    target_width = max(round(w * target_height / h), 1)
    if max_width is not None and target_width > max_width:
        target_width = max_width
    image = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
    return image
