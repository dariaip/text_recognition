from abc import ABC, abstractmethod
from typing import List, Any, Dict

import cv2
import numpy as np
from torch.utils.data import Dataset

from .item import Item


class BaseAdapter(Dataset, ABC):
    def __init__(
        self,
        unclip_ratio: float = 0,
        fit_min_rot_box: bool = False,
        in_memory: bool = False
    ):
        super(BaseAdapter, self).__init__()

        self.in_memory = in_memory
        self.unclip_ratio = unclip_ratio
        self.fit_min_rot_box = fit_min_rot_box

        self.annotations: List[Any] = []
        self.image_fpaths: List[str] = []
        self.load_anns()
        self._prepare_anns()

        self.images: Dict = {}
        if in_memory:
            self.load_images()

        if len(self.annotations) == 0:
            raise Exception('Annotation list is empty!')

        if len(self.image_fpaths) == 0:
            raise Exception('FName list is empty!')

    @abstractmethod
    def load_anns(self) -> None:
        pass

    @abstractmethod
    def _prepare_anns(self) -> None:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Item:
        pass

    def get_ann(self, idx: int) -> Any:
        """
        Вспомогательный метод для загрузки аннотации по индексу.

        Args:
            idx: индекс необходимой аннотации

        Returns:
            аннотация
        """
        return self.annotations[idx]

    def load_images(self) -> None:
        """
        Вспомогательный для загрузки изображений из всех аннотаций в память.
        """
        self.images = {}
        for idx in range(len(self)):
            img = self.load_image(idx)
            self.images[idx] = img

    def load_image(self, idx: int) -> np.ndarray:
        """
        Вспомогательный метод для загрузки одного изображения по индексу.

        Args:
            idx: индекс изображения

        Returns:
            загруженное изображение
        """
        fname = self.image_fpaths[idx]
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_image(self, idx: int) -> np.ndarray:
        """
        Вспомогательный метод для загрузки изображения из памяти, если self.in_memory является истинным, иначе
        с диска.

        Args:
            idx: индекс изображение

        Returns:
            загруженное изображение
        """
        if self.in_memory:
            return self.images[idx]
        return self.load_image(idx)

    def __len__(self) -> int:
        """
        Метод для подсчета количества аннотаций в данном адаптере.

        Returns:
            количество аннотаций
        """
        return len(self.annotations)
