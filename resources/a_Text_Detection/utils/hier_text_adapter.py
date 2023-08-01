import json
import os

import numpy as np

from .base_adapter import BaseAdapter
from .item import Item
from .unclip import unclip


class HierTextAdapter(BaseAdapter):
    def __init__(
        self,
        img_dir: str,
        ann_path: str,
        unclip_ratio: float = 0,
        line_unclip_ratio: float = 0,
        group_unclip_ratio: float = 0,
        fit_min_rot_box: bool = False,
        in_memory: bool = False
    ):
        self.ann_path = ann_path
        self.img_dir = img_dir
        self.black_polygons = []
        self.line_unclip_ratio = line_unclip_ratio
        self.group_unclip_ratio = group_unclip_ratio

        super().__init__(
            unclip_ratio=unclip_ratio,
            fit_min_rot_box=fit_min_rot_box,
            in_memory=in_memory
        )

    def load_anns(self):
        """
        Вспомогательный метод для  загрузки аннотаций из JSON файла с разметкой.
        """
        self.image_fpaths = []

        self.annotations = []
        with open(self.ann_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        img_names = []
        for anno in annotations['annotations']:
            img_name = anno['image_id'] + '.jpg'
            img_names.append(img_name)
            self.image_fpaths.append(os.path.join(self.img_dir, img_name))

            polygons, words, char_polygons, chars, line_polygons, group_polygons = self.prepare_ann(anno)
            ann = [polygons, words, char_polygons, chars, line_polygons, group_polygons]

            self.annotations.append(ann)

    def _prepare_anns(self):
        """
        Вспомогательный метод для подготовки всех аннотаций.
        """
        out_anns = []
        for word_polygons, words, char_polygons, chars, line_polygons, group_polygons in self.annotations:
            if self.unclip_ratio != 0:
                word_polygons = unclip(word_polygons, self.unclip_ratio)
                line_polygons = unclip(line_polygons, self.line_unclip_ratio)
                group_polygons = unclip(group_polygons, self.group_unclip_ratio)

            if self.fit_min_rot_box:
                word_polygons = fit_min_rot_box(word_polygons)

            out_anns.append((word_polygons, words, char_polygons, chars, line_polygons, group_polygons))
        self.annotations = out_anns

    @staticmethod
    def prepare_ann(ann):
        """
        Вспомогательный метод для подготовки всех необходимых данных из одной аннотации.

        Args:
            ann: аннотация

        Returns:
            кортеж со списком полигонов текста, списком слов, списком полигонов символов, списком символов,
                списком полигонов линий и списком полигонов групп
        """
        polygons = []
        line_polygons = []
        group_polygons = []
        words = []
        for par in ann['paragraphs']:
            bbox = np.array(par['vertices']).astype(int)

            group_polygons.append(bbox)

            for line in par['lines']:
                bbox = np.array(line['vertices']).astype(int)

                line_polygons.append(bbox)

                for word in line['words']:
                    bbox = np.array(word['vertices']).astype(int)

                    polygons.append(bbox)

        char_polygons, chars = None, None
        return polygons, words, char_polygons, chars, line_polygons, group_polygons

    def __getitem__(self, idx) -> Item:
        """
        Основной метод, который будет вызываться извне и возвращать объект типа Item со всеми данными об изображении.

        Args:
            idx: индекс необходимого сэмпла

        Returns:
            сэмлп с данными об изображении
        """
        img = self.get_image(idx)

        word_polygons, words, char_polygons, chars, line_polygons, group_polygons = self.get_ann(idx)

        return Item(
            fname=self.image_fpaths[idx],
            img=img,
            word_polygons=word_polygons,
            words=words,
            char_polygons=char_polygons,
            chars=chars,
            line_polygons=line_polygons,
            group_polygons=group_polygons,
        )
