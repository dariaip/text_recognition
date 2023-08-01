from typing import List, Tuple

import cv2
import numpy as np


class DrawMore:
    """
    Вспомогательный класс с набором методов для конвертации и отрисовки.
    """
    @staticmethod
    def draw_contours(
        image: np.ndarray,
        polygons: List[np.ndarray],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        fill: bool = False,
        inplace: bool = False
    ) -> np.ndarray:
        """
        Вспомогательный метод для отрисовки набора контуров (полигонов) на изображении.

        Args:
            image: изображение
            polygons: список полигонов
            color: цвет отрисовки полигонов
            thickness: толщина отрисовки полигонов
            fill: заливать ли отрисованные полигоны цветом
            inplace: изменять ли исходное изображение

        Returns:
            исходное изображение, если inplace=True, иначе копия исходного изображения с отрисованными полигонами
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
