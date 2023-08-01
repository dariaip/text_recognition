from typing import Tuple, List

import cv2
from tqdm.auto import tqdm

from .csv_adapter import CsvAdapter


def get_shapes_and_texts(adapter: CsvAdapter) -> Tuple[List[str], List[int], List[int]]:
    """
    Вспомогательный метод для загрузки всех размерностей и текстов из адаптера.
    
    Args:
        adapter: адаптер типа CsvAdapter
    Returns:
        список всех текстов, список ширины всех изображений, список длины всех изображений
    """
    all_items = [adapter.__getitem__(i) for i in range(len(adapter))]
    all_texts = [item.label for item in all_items]
    all_images = [cv2.cvtColor(cv2.imread(item.image_fpath), cv2.COLOR_BGR2RGB) for item in tqdm(all_items)]
    all_heights = [image.shape[0] for image in all_images]
    all_widths = [image.shape[1] for image in all_images]
    return all_texts, all_widths, all_heights
