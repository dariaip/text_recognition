from typing import Tuple, List

import cv2
import numpy as np

from .unclip_polygon import unclip_polygon


class Postprocessor:
    def __init__(
        self,
        binarization_threshold: float = 0.3,
        confidence_threshold: float = 0.7,
        unclip_ratio: float = 1.5,
        min_area: float = 0,
        max_number: int = 1000
    ):
        """
        Класс для постобработки полигонов.

        Args:
            binarization_threshold: граница для бинаризации
            confidence_threshold: граница по уверенности, ниже которой полигоны будут удаляться
            unclip_ratio: значение для расширения полигонов
            min_area: минимальная площадь, ниже которой полигоны будут удаляться
            max_number: максимальное количество полигонов
        """
        self.binarization_threshold = binarization_threshold
        self.confidence_threshold = confidence_threshold
        self.unclip_ratio = unclip_ratio
        self.min_area = min_area
        self.max_number = max_number

    def __call__(
        self,
        width: int,
        height: int,
        pred: np.ndarray,
        return_polygon: bool = False
    ) -> Tuple[List[List[np.ndarray]], List[List[float]]]:
        """
        Основной метод класса, принимающий на вход предсказания модели и возвращающий прямоугольники/полигоны
        (в зависимости от значения параметра return_polygon).

        Args:
            width: ширина исходного изображения
            height: высота исходного изображения
            pred: предсказания модели
            return_polygon: если True, то возвращаются полигоны, иначе возвращаются прямоугольники

        Returns:
            прямоугольники/полигоны на основе предсказаний модели
        """
        pred = pred[:, 0, :, :]
        segmentation = self.binarize(pred)
        boxes_batch = []
        scores_batch = []
        for batch_index in range(pred.shape[0]):
            if return_polygon:
                boxes, scores = self.polygons_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            else:
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def binarize(self, pred: np.ndarray) -> np.ndarray:
        """
        Вспомогательный метод для бинаризации (приведения тензора к значениям только 0 или 1) предсказаний по порогу
        self.binarization_threshold.

        Args:
            pred: предсказания модели

        Returns:
            бинаризованные предсказания модели
        """
        return pred > self.binarization_threshold

    def polygons_from_bitmap(
        self,
        pred: np.ndarray,
        _bitmap: np.ndarray,
        dest_width: int,
        dest_height: int
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Вспомогательный метод для создания полигонов из бинаризованных предсказаний модели. Все предсказанные полигоны
        приводятся к размеру исходного изображения (dest_width, dest_height).

        Args:
            pred: предсказания модели
            _bitmap: бинаризованные предсказания модели
            dest_width: ширина исходного изображения
            dest_height: высота исходного изображения

        Returns:
            полигоны на основе предсказаний модели
        """
        assert len(_bitmap.shape) == 2
        height, width = _bitmap.shape
        contours = self._get_good_contours(_bitmap)
        boxes = []
        scores = []

        for contour in contours:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = self.box_score_fast(pred, contour.squeeze(1))
            if score < self.confidence_threshold:
                continue

            if points.shape[0] > 2:
                box = unclip_polygon(points, unclip_ratio=self.unclip_ratio)
                if len(box) > 1:
                    continue

                if len(box) == 0:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)

            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.round(box[:, 0] / width * dest_width)
            box[:, 1] = np.round(box[:, 1] / height * dest_height)

            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(
        self,
        pred: np.ndarray,
        _bitmap: np.ndarray,
        dest_width: int,
        dest_height: int
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Вспомогательный метод для создания прямоугольников (ббоксов) из бинаризованных предсказаний модели.
        Все предсказанные прямоугольники приводятся к размеру исходного изображения (dest_width, dest_height).

        Args:
            pred: предсказания модели
            _bitmap: бинаризованные предсказания модели
            dest_width: ширина исходного изображения
            dest_height: высота исходного изображения

        Returns:
            прямоугольники на основе предсказаний модели
        """
        assert len(_bitmap.shape) == 2
        height, width = _bitmap.shape
        contours = self._get_good_contours(_bitmap)
        num_contours = len(contours)
        boxes = []
        scores = []

        for index in range(num_contours):
            contour = contours[index].squeeze(1)

            score = self.box_score_fast(pred, contour)
            if score < self.confidence_threshold:
                continue

            points, sside = self.get_mini_boxes(contour)
            points = np.array(points)
            box = unclip_polygon(points, unclip_ratio=self.unclip_ratio).reshape((-1, 1, 2))

            if len(box) < 1:
                continue

            box, sside = self.get_mini_boxes(box)
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.round(box[:, 0] / width * dest_width)
            box[:, 1] = np.round(box[:, 1] / height * dest_height)

            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def _get_good_contours(self, _bitmap: np.ndarray) -> List[np.ndarray]:
        """
        Вспомогательный метод для нахождения всех контуров на изображении, имеющих площадь больше self.min_area.
        Возвращаются максимум self.max_number самых больших по площади контуров.

        Args:
            _bitmap: бинаризованные предсказания модели

        Returns:
            self.max_number самых больших по площади контуров, имеющих площадь больше self.min_area
        """
        contours, _ = cv2.findContours((_bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_areas = [cv2.contourArea(c) for c in contours]
        contours = [c
                    for c, a in sorted(zip(contours, contour_areas), key=lambda pair: pair[1], reverse=True)
                    if a >= self.min_area][:self.max_number]
        return contours

    @staticmethod
    def get_mini_boxes(contour: np.ndarray) -> Tuple[List[List[int]], int]:
        """
        Вспомогительный метод, приводящий заданный контур к списку из 4 точек с помощью метода cv2.minAreaRect.

        Args:
            contour: контур

        Returns:
            список из 4 точек, длина меньшей стороны результирующего прямоугольника
        """
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    @staticmethod
    def box_score_fast(bitmap: np.ndarray, _box: np.ndarray) -> float:
        """
        Вспомогательный метод для оценки уверенности модели в заданном контуре. Уверенность вычисляется как процент
        значений 1 в бинаризованном предсказании модели в пределах заданного контура.

        Args:
            bitmap: бинаризованные предсказания модели
            _box: контур

        Returns:
            уверенность модели в заданном контуре
        """
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(int), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
