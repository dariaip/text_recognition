from pathlib import Path
from typing import Optional, List, Union

import matplotlib.pyplot as plt
import numpy as np

from .make_label import make_label


def draw_predictions(
    images: List[np.ndarray],
    target_texts: Optional[List[str]] = None,
    predicted_texts: Optional[List[str]] = None,
    path_to_save_image: Optional[Union[str, Path]] = None,
    columns: int = 4,
    max_elements_to_draw: int = 64,
    fontsize: int = 10,
) -> Optional[plt.Figure]:
    """
    Функция для визуализации изображений с предсказанными моделью / истинными текстами.

    Args:
        images: изображения
        target_texts: истинные тексты
        predicted_texts: предсказанные моделью тексты
        path_to_save_image: опциональный путь для сохранения итогового изображения
        columns: количество столбцов на итоговом изображении
        max_elements_to_draw: максимальное количество отображаемых изображений
        fontsize: размер шрифта

    Returns:
        None, если задан параметр path_to_save_image, иначе изображение в виде plt.Figure
    """
    if target_texts is None:
        target_texts = [None] * len(images)
    if predicted_texts is None:
        predicted_texts = [None] * len(images)

    if len(images) == 1:
        fig = plt.figure(figsize=(3, 3))
        plt.xlabel(make_label(predicted_texts[0], target_texts[0]), fontsize=fontsize)
        plt.imshow(images[0].astype(np.uint8), vmin=0, vmax=255)
    else:
        rows = min(max_elements_to_draw, len(target_texts)) // columns + 1
        fig = plt.figure(figsize=(20, int(rows * 2)))
        for i, (target_text, pred_text, image) in enumerate(zip(target_texts, predicted_texts, images)):
            if i + 1 > rows * columns:
                break
            ax = fig.add_subplot(rows, columns, i + 1)
            plt.xlabel(make_label(pred_text, target_text), fontsize=fontsize)
            ax.set_facecolor("white")
            plt.imshow(image.astype(np.uint8), vmin=0, vmax=255)

    if path_to_save_image is not None:
        fig.savefig(path_to_save_image, bbox_inches='tight')
        plt.close(fig)
        return None
    else:
        return fig
