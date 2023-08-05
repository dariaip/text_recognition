from typing import Tuple, List

import cv2
import numpy as np

from resources.unclip_polygon import unclip_polygon


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
        Class for polygons' postprocessing.

        Args:
            binarization_threshold: threshold for binarization
            confidence_threshold: threshold for confidence (a polygons with a lower confidence will be removed)
            unclip_ratio: value for expanding polygons
            min_area: minimal area (a polygons with a smaller will be removed)
            max_number: maximal number of polygons
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
        The main method of the class

        Args:
            width: width of the initial image
            height: height of the initial image
            pred: model's predictions
            return_polygon: if True, than polygons will be returned, rectangles will be returned otherwise

        Returns:
            rectangles/polygons based on the model's predictions
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
        Auxiliary method for predoctions' binarization (the tensor with predictions will includes 1 or 0 values) according to the self.binarization_threshold.

        Args:
            pred: model's predictions

        Returns:
            binarized predictions
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
        Auxiliary method for polygons' creation from the provided binarized predictions.
        All predictied polygons are scaled to the size of the initial image (dest_width, dest_height).

        Args:
            pred: model's predictions
            _bitmap: binarized predictions
            dest_width: width of the initial image
            dest_height: height of the initial image

        Returns:
            polygons
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
        Auxiliary method for creating bounding boxes (bboxes) from binarized model's predictions.
        All predicted rectangles are scaled to a size of the initial image (dest_width, dest_height).

        Args:
            pred: model's predictions
            _bitmap: binarized model's predictions
            dest_width: width of the initial image
            dest_height: height of the initial image

        Returns:
            bboxes
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
        Auxiliary method for finding all contours on an image that has bigger area than self.min_area.

        Args:
            _bitmap: binarized model's predictions

        Returns:
            self.max_number of the biggest contours (according to their area) that has bigger area than self.min_area
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
        Auxiliary method that transforms a contour to a list of 4 points with help of cv2.minAreaRect method.

        Args:
            contour: contour

        Returns:
            a list of 4 points and a length of the shortest side of the resulting rectangle
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
        Auxiliary method for estimation of model's confidence in the provided contour.
        The confidence is equal to a percent of ones into the binarized predictions inside the provided contour.

        Args:
            bitmap: the binarized predictions
            _box: a contour

        Returns:
            model's confidence inside the contour
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
