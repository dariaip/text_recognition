from typing import Optional

import cv2
import numpy as np


def resize_by_height(image: np.ndarray, target_height: int, max_width: Optional[int] = None) -> np.ndarray:
    """
    Вспомогательный метод для изменения размера изображения по высоте.
    Итоговый размер определяется на основе высоты target_height, а ширина будет не более max_width,
    если параметр max_width указан.

    Args:
        image: входное изображение
        target_height: целевая высота, к которой приведется изображение
        max_width: (опционально) максимальная возможная ширина преобразованного изображения

    Returns:
        преобразованное изображение
    """
    h, w, _ = image.shape
    interpolation = cv2.INTER_AREA if h > target_height else cv2.INTER_LINEAR
    target_width = max(round(w * target_height / h), 1)
    if max_width is not None and target_width > max_width:
        target_width = max_width
    image = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
    return image
