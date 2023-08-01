import random
import warnings
from typing import List, Tuple, Union, Any

import cv2
import numpy as np
import torch
from albumentations import Compose, OneOf, BasicTransform
from torch.utils.data import Dataset

from .csv_adapter import CsvAdapter
from .item import Item
from .resize_by_height import resize_by_height


class OCRDataset(Dataset):
    def __init__(
        self,
        adapters: List[Tuple[CsvAdapter, float]],
        transforms: Union[BasicTransform, Compose, OneOf],
        post_transforms: Union[BasicTransform, Compose, OneOf],
        tokenizer: Any,
        epoch_size: int = -1,
        sort_by_length: bool = True,
        target_height: int = 32,
        max_width: int = 1024,
    ):
        """
        Класс датасета OCR.

        Args:
            adapters: список адаптеров типа CsvAdapter с вероятностями каждого адаптера (вероятности
                должны суммироваться к 1)
            transforms: необязательные аугментации из albumentations
            post_transforms: обязательные преобразования (нормализация, приведение к тензору)
            tokenizer: токенайзер, объект класса TokenizerForCTC
            epoch_size: размер эпохи (если передано число меньше 0, будут использованы все данные)
            sort_by_length: использовать ли сортировку батчей по ширине изображения / длине текста
            target_height: высота, к которой будут приводиться все изображения
            max_width: максимальная ширина изображения (если ширина будет больше, то изображение будет 
            уменьшено по ширине до max_width)
        """
        self.adapters = adapters
        self.transforms = transforms
        self.post_transforms = post_transforms
        if self.transforms is None:
            warnings.warn("You did not choose any transforms.", UserWarning)

        self.tokenizer = tokenizer
        self.sort_by_length = sort_by_length
        ratio = sum([r for _, r in self.adapters])
        if ratio != 1.:
            raise ValueError(f'Sum of adapter ratios must be 1, but is {ratio}')

        self.epoch_size = epoch_size
        if epoch_size < 0:
            self.epoch_size = sum([len(a) for a, _ in self.adapters])

        print('Epoch size: ', epoch_size)
        print('Adapters')
        for adapter, ratio in self.adapters:
            print('Adapter name: ', adapter.name)
            print('Adapter epoch ratio: ', ratio)
            print('Adapter epoch size: ', int(ratio * self.epoch_size))

        self.target_height = target_height
        self.max_width = max_width

        self.epoch_data = self._prepare_epoch_data()

    def _prepare_epoch_data(self) -> List[Any]:
        """
        Вспомогательный метод для подготовки данных для всей эпохи.

        Returns:
            список из self.epoch_size сэмплов
        """
        data = []
        for adapter, ratio in self.adapters:
            if self.epoch_size < 0:
                size = len(adapter)
            else:
                size = int(self.epoch_size * ratio)

            idxs = random.sample(range(0, len(adapter)), k=min(len(adapter), size))
            idxs += random.choices(range(0, len(adapter)), k=size - len(idxs))
            data += [adapter[idx] for idx in idxs]

        if self.sort_by_length:
            data = sorted(data, key=lambda item: item.length)

        return data

    @staticmethod
    def _get_image(sample: Item) -> np.ndarray:
        """
        Вспомогательный метод для загрузки изображения из объекта типа Item или из пути

        Args:
            sample: сэмпл

        Returns:
            загруженное изображение
        """
        if sample.image is not None:
            return sample.image
        image = cv2.imread(sample.image_fpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def reset_epoch(self) -> None:
        """
        Метод для обновления данных для новой эпохи.
        """
        self.epoch_data = self._prepare_epoch_data()

    # убрать!
    def _prepare_sample(self, sample: Item) -> Tuple[Any, np.ndarray, str, torch.Tensor, int]:
        """
        Вспомогательный метод для подготовки сэмпла:
        - загрузка изображения и текста
        - токенизация текста
        - применение опциональных аугментаций
        - применение обязательных преобразований изображения.

        Args:
            sample: сэмлп

        Returns:
            преобразованное изображение; исходное изображение; текст; токенизированный текст; длина текста
        """
        raw_image = self._get_image(sample)
        raw_image = resize_by_height(raw_image, self.target_height, self.max_width)
        label = sample.label
        target, length = self.tokenizer.encode(label)

        if self.transforms:
            raw_image = self.transforms(image=raw_image)['image']

        image = self.post_transforms(image=raw_image)['image']

        return image, raw_image, label, target, length

    # убрать!
    def _get_sample(self, idx: int) -> Any:
        """
        Вспомогательный метод для извлечения нужного сэмпла по индексу.

        Args:
            idx: индекс требуемого сэмпла

        Returns:
            сэмпл
        """
        return self.epoch_data[idx]

    def __getitem__(self, idx: int) -> Tuple:
        """
        Основной метод, который вызывается извне и возвращать преобразованный сэмлп в развернутом виде.

        Args:
            idx: индекс требуеиого сэмпла

        Returns:
            преобразованный сэмпл в развернутом виде
        """
        sample = self._get_sample(idx)
        result = self._prepare_sample(sample)
        return result

    def __len__(self) -> int:
        """
        Метод для подсчета количества сэмплов в одной эпохе.

        Returns:
            количество сэмплов в одной эпохе
        """
        if self.epoch_size < 0:
            return sum([len(a) for a, _ in self.adapters])
        return self.epoch_size
