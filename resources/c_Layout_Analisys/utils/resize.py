from typing import Tuple

import cv2
import numpy as np


def resize_aspect_ratio(
    img: np.ndarray, square_size: int, interpolation: int, mag_ratio: int = 1
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    height, width, channel = img.shape

    target_size = mag_ratio * max(height, width)

    if target_size > square_size:
        target_size = square_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w / 2), int(target_h / 2))

    return resized, ratio, size_heatmap
