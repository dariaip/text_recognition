from typing import Dict, Union, List

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

np.seterr(divide='ignore', invalid='ignore')


class MakeBorderMap:
    def __init__(self, shrink_ratio: float = 0.4, thresh_min: float = 0.3, thresh_max: float = 0.7):
        """


        Args:
            shrink_ratio:
            thresh_min:
            thresh_max:
        """
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max

    def __call__(self, data: Dict) -> Dict:
        """


        Args:
            data:

        Returns:

        """
        im = data['img']
        text_polys = data['text_polys']
        ignore_tags = data['ignore_tags']

        canvas = np.zeros(im.shape[:2], dtype=np.float32)
        mask = np.zeros(im.shape[:2], dtype=np.float32)

        for i in range(len(text_polys)):
            if ignore_tags[i]:
                continue
            self.draw_border_map(text_polys[i], canvas, mask=mask)
        canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min

        data['threshold_map'] = canvas
        data['threshold_mask'] = mask
        return data

    def draw_border_map(self, polygon: Union[List, np.ndarray], canvas: np.ndarray, mask: np.ndarray) -> None:
        """


        Args:
            polygon:
            canvas:
            mask:

        Returns:

        """
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        if polygon_shape.area <= 0:
            return
        distance = polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

        padded_polygon = np.array(padding.Execute(distance)[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros(
            (polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid - ymin:ymax_valid - ymax + height,
                xmin_valid - xmin:xmax_valid - xmax + width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

    @staticmethod
    def distance(xs: np.ndarray, ys: np.ndarray, point_1: np.ndarray, point_2: np.ndarray) -> np.ndarray:
        """
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        """
        sq_dist_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        sq_dist_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        sq_dist = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (sq_dist - sq_dist_1 - sq_dist_2) / (2 * np.sqrt(sq_dist_1 * sq_dist_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)

        result = np.sqrt(sq_dist_1 * sq_dist_2 * square_sin / sq_dist)
        result[cosin < 0] = np.sqrt(np.fmin(sq_dist_1, sq_dist_2))[cosin < 0]
        return result
