import json
import os
from typing import Any, Tuple

from .base_adapter import BaseAdapter
from .fit_min_rot_box import fit_min_rot_box
from .item import Item
from .unclip import unclip


class LabelStudioAdapter(BaseAdapter):
    def __init__(
        self,
        img_dir: str,
        ann_path: str,
        unclip_ratio: float = 0,
        fit_min_rot_box: bool = False,
        in_memory: bool = False
    ):
        """
        Класс адаптера для разметки детекции текста.

        Args:
            img_dir: путь к папке с изображениям
            ann_path: путь к файлу с разметкой
            unclip_ratio: значение расширения полигонов
            fit_min_rot_box: приводить ли полигоны к виду повернутого прямоугольника
            in_memory: загружать ли изображения в память при инициализации адаптера
        """
        self.ann_path = ann_path
        self.img_dir = img_dir
        super().__init__(unclip_ratio, fit_min_rot_box, in_memory)

    def load_anns(self) -> None:
        """
        Вспомогательный метод для  загрузки аннотаций из JSON файла с разметкой.
        """
        self.image_fpaths = []

        self.annotations = []
        with open(self.ann_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        img_names = []
        for anno in annotations:
            img_name = anno['data']['image'].split('/')[-1]
            img_names.append(img_name)
            self.image_fpaths.append(os.path.join(self.img_dir, img_name))

            polygons, words, char_polygons, chars = self.prepare_ann(anno)
            ann = [polygons, words, char_polygons, chars]

            self.annotations.append(ann)

    def _prepare_anns(self) -> None:
        """
        Вспомогательный метод для подготовки всех аннотаций.
        """
        out_anns = []
        for word_polygons, words, char_polygons, chars in self.annotations:
            if self.unclip_ratio != 0:
                word_polygons = unclip(word_polygons, self.unclip_ratio)

            if self.fit_min_rot_box:
                word_polygons = fit_min_rot_box(word_polygons)

            out_anns.append((word_polygons, words, char_polygons, chars))
        self.annotations = out_anns

    @staticmethod
    def prepare_ann(ann: Any) -> Tuple:
        """
        Вспомогательный метод для подготовки всех необходимых данных из одной аннотации.

        Args:
            ann: аннотация

        Returns:
            кортеж со списком полигонов текста, списком слов, списком полигонов символов и списком символов
        """
        polygons = []
        black_polygons = []
        # -1 get last of an available annotation
        for elem in ann['annotations'][-1]['result']:
            label = elem['value']

            if elem['type'] in ['choices']:
                continue
            # for unscaling to original size from LabelStudio's 0-100 range
            width_scale = elem['original_width'] / LS_SIDE_LIMIT
            height_scale = elem['original_height'] / LS_SIDE_LIMIT
            if POLYGON_FIELDS.issubset(set(label.keys())):
                # polygon label case
                polygon = label['points']
                polygon = np.array([[
                    p[0] * width_scale,
                    p[1] * height_scale
                ] for p in polygon]).astype(int)

                # check if this polygon label contains valid word label
                if contain_any(VALID_LABELS, label['polygonlabels']):
                    polygons.append(polygon)
                # check if this polygon should be blacked
                if contain_any(BLACK_LABELS, label['polygonlabels']):
                    black_polygons.append(polygon)

            if RECTANGLE_FIELDS.issubset(set(label.keys())):  # bbox label case
                # unscale to original size from LabelStudio's 0-100 range
                bbox = (label['x'], label['y'], label['width'], label['height'])
                bbox = scale_bbox(bbox, width_scale, height_scale)
                polygon = bbox_to_polygon(bbox)

                # check if bbox is rotated
                if 'rotation' in label.keys() and label['rotation'] != 0:
                    center = polygon[0]
                    polygon = rotate(polygon, center, degree=-label['rotation']).astype(int)

                # check if this bbox label contains valid word label
                if contain_any(VALID_LABELS, label['rectanglelabels']):
                    polygons.append(polygon)
                # check if this bbox should be blacked
                elif contain_any(BLACK_LABELS, label['rectanglelabels']):
                    black_polygons.append(polygon)

        words, char_polygons, chars = None, None, None
        return polygons, words, char_polygons, chars

    def __getitem__(self, idx: int) -> Item:
        """
        Основной метод, который будет вызываться извне и возвращать объект типа Item со всеми данными об изображении.

        Args:
            idx: индекс необходимого сэмпла

        Returns:
            сэмлп с данными об изображении
        """
        img = self.get_image(idx)

        word_polygons, words, char_polygons, chars = self.get_ann(idx)

        return Item(
            fname=self.image_fpaths[idx],
            img=img,
            word_polygons=word_polygons,
            words=words,
            char_polygons=char_polygons,
            chars=chars
        )
