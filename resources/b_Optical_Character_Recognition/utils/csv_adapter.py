import csv
import os
from typing import Tuple, Optional

import cv2
import numpy as np

from .item import Item


class CsvAdapter(object):
    def __init__(
        self,
        name: str,
        images_dir: str,
        csv_fpath: str,
        compute_length_by: str = "label",  # label, image
        load_image: bool = False
    ):
        """
        Класс адаптера для разметки OCR.

        Args:
            name: имя адаптера
            images_dir: путь к папке с изображениям
            csv_fpath: путь к разметке в формате CSV
            compute_length_by: "label": считать длину по длину текста; "image": считать длину по ширине изображения
            load_image: загружать ли изображения в память при инициализации
        """
        self.name = name
        self.images_dir = images_dir
        self.csv_fpath = csv_fpath
        self.compute_length_by = compute_length_by
        self.load_image = load_image

        self.annotations = None

        self._load_anns()

    def _load_anns(self) -> None:
        """
        Вспомогательный метод для  загрузки аннотаций из CSV файла с разметкой.
        """
        anns = []
        with open(self.csv_fpath, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            _ = next(reader)
            for fname, label in reader:
                anns.append((os.path.join(self.images_dir, fname), label))
        self.annotations = anns

    def _get_ann(self, idx: int) -> Tuple[str, str]:
        """
        Вспомогательный метод для загрузки аннотации по индексу.

        Args:
            idx: индекс необходимой аннотации

        Returns:
            аннотация
        """
        image_fpath, label = self.annotations[idx]
        return image_fpath, label

    @staticmethod
    def _load_image(fname: str) -> np.ndarray:
        """
        Вспомогательный метод для загрузки изображения с диска.

        Args:
            fname: путь к изображению

        Returns:
            загруженное с диска изображение
        """
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def _get_image_width(image: Optional[np.ndarray], fname: str) -> int:
        """
        Вспомогательный метод для определения ширины изображения.

        Args:
            image: изображение (может быть None)
            fname: путь к изображению, по которому загрузится изображение и определится ширина

        Returns:
            ширина изображения
        """
        if image is not None:
            return image.shape[1]
        loaded_image = cv2.imread(fname)
        return loaded_image.shape[1]

    def __len__(self) -> int:
        """
        Метод для подсчета количества аннотаций в данном адаптере.

        Returns:
            количество аннотаций
        """
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Item:
        """
        Основной метод, который будет вызываться извне и возвращать объект типа Item со всеми данными об изображении.

        Args:
            idx: индекс необходимого сэмпла

        Returns:
            сэмлп с данными об изображении
        """
        image = None
        image_fpath, label = self._get_ann(idx)
        if self.load_image:
            image = self._load_image(image_fpath)

        if self.compute_length_by == 'label':
            length = len(label.strip())
        elif self.compute_length_by == 'image':
            length = self._get_image_width(image, image_fpath)
        else:
            raise ValueError(f'Unsupported compute_length_by value: {self.compute_length_by}')

        return Item(
            image_fpath=image_fpath,
            image=image,
            label=label,
            length=length
        )
