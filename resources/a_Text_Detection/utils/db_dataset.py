import warnings
from typing import List, Union, Tuple

import numpy as np
from albumentations import BasicTransform, Compose, OneOf
from torch.utils.data import Dataset

from .base_adapter import BaseAdapter
from .item import Item
from .make_border_map import MakeBorderMap
from .make_shrink_map import MakeShrinkMap


class DBDataset(Dataset):
    def __init__(
        self,
        adapters: List[BaseAdapter],
        transforms: Union[BasicTransform, Compose, OneOf],
        post_transforms: Union[BasicTransform, Compose, OneOf],
        shrink_ratio: float = 0.6,
        size: int = -1
    ):
        """
        Класс датасета для детекции текста.

        Args:
            adapters: список адаптеров типа BaseAdapter
            transforms: необязательные аугментации из albumentations
            post_transforms: обязательные преобразования (нормализация, приведение к тензору)
            shrink_ratio: коэффициент сжатия полигонов,
            size: размер датасета (в картинках, не в батчах)
        """
        self.adapters = adapters
        self.transforms = transforms
        self.post_transforms = post_transforms
        if self.transforms is None:
            warnings.warn("You did not choose any transforms.", UserWarning)

        self.border_creator = MakeBorderMap(
            shrink_ratio=shrink_ratio,
            thresh_min=0.3,
            thresh_max=0.7,
        )
        self.mask_shrinker = MakeShrinkMap(
            shrink_ratio=shrink_ratio,
            min_text_size=3.,
            shrink_type='pyclipper',
        )
        self.size = size

    def __getitem__(self, idx: int) -> Tuple:
        """
        Основной метод, который будет вызываться извне и возвращать преобразованный сэмлп в развернутом виде.

        Args:
            idx: индекс требуеиого сэмпла

        Returns:
            преобразованный сэмпл в развернутом виде
        """
        forward_idx = idx % self.__len__()
        for adapter in self.adapters:
            if forward_idx - len(adapter) + 1 <= 0:
                sample = adapter[forward_idx]
            else:
                forward_idx -= len(adapter)

        targets = self.prepare_sample(sample)
        return targets

    def __len__(self) -> int:
        """
        Метод для подсчета количества сэмплов в одной эпохе.

        Returns:
            количество сэмплов в одной эпохе
        """
        size_from_adapters = sum([len(a) for a in self.adapters])
        if self.size != -1:
            return min(self.size, size_from_adapters)
        return size_from_adapters

    def prepare_sample(self, sample: Item) -> Tuple:
        """
        Вспомогательный метод для подготовки сэмпла:
        - загрузка изображений и полигонов
        - подготовка масок для обучения из полигонов
        - применение опциональных аугментаций
        - применение обязательных преобразований изображения.

        Args:
            sample: сэмпл

        Returns:
            исходное изображение, преобразованное изображение, ...
        """
        raw_img = sample.img
        word_polygons = sample.word_polygons

        res = self.mask_shrinker({
            'img': raw_img,
            'text_polys': word_polygons,
            'ignore_tags': [0] * len(word_polygons)
        })
        shrink_map = res['shrink_map']
        shrink_mask = res['shrink_mask']

        res = self.border_creator({
            'img': raw_img,
            'text_polys': word_polygons,
            'ignore_tags': [0] * len(word_polygons)
        })
        threshold_map = res['threshold_map']
        threshold_mask = res['threshold_mask']

        transformed = self.transforms(
            image=raw_img,
            masks=[shrink_map, shrink_mask, threshold_map, threshold_mask]
        )
        raw_img = transformed['image']
        shrink_map, shrink_mask, threshold_map, threshold_mask = transformed['masks']

        post_transformed = self.post_transforms(
            image=raw_img,
            masks=[shrink_map, shrink_mask, threshold_map, threshold_mask]
        )
        img = post_transformed['image']
        shrink_map, shrink_mask, threshold_map, threshold_mask = post_transformed['masks']
        shrink_mask = np.full_like(shrink_mask, fill_value=1.)

        return raw_img, img, shrink_map, shrink_mask, threshold_map, threshold_mask, word_polygons
