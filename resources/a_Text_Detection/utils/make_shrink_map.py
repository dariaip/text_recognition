from typing import Dict, List, Tuple

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

np.seterr(divide='ignore', invalid='ignore')


class MakeShrinkMap:
    def __init__(self, min_text_size: float = 1, shrink_ratio: float = 0.4, shrink_type: str = 'pyclipper'):
        """


        Args:
            min_text_size:
            shrink_ratio:
            shrink_type:
        """
        assert shrink_type in ['py', 'pyclipper'], "параметр shrink_type должен быть одним из ['py','pyclipper']"
        shrink_func_dict = {'py': self.shrink_polygon_py, 'pyclipper': self.shrink_polygon_pyclipper}
        self.shrink_func = shrink_func_dict[shrink_type]
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio

    def __call__(self, data: Dict) -> Dict:
        """


        Args:
            data:

        Returns:

        """
        image = data['img']
        text_polys = data['text_polys']
        ignore_tags = data['ignore_tags']

        h, w = image.shape[:2]
        text_polys, ignore_tags = self.validate_polygons(text_polys, ignore_tags, h, w)
        gt = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(text_polys)):
            polygon = text_polys[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                shrinked = self.shrink_func(polygon, self.shrink_ratio)
                if shrinked.size == 0:
                    cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue
                cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)

        data['shrink_map'] = gt
        data['shrink_mask'] = mask
        return data

    @staticmethod
    def validate_polygons(
        polygons: List[np.ndarray],
        ignore_tags: List[bool],
        h: int,
        w: int
    ) -> Tuple[List[np.ndarray], List[bool]]:
        """


        Args:
            polygons:
            ignore_tags:
            h:
            w:

        Returns:

        """
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = cv2.contourArea(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    @staticmethod
    def shrink_polygon_py(polygon: np.ndarray, shrink_ratio: float) -> np.ndarray:
        """


        Args:
            polygon:
            shrink_ratio:

        Returns:

        """
        cx = polygon[:, 0].mean()
        cy = polygon[:, 1].mean()
        polygon[:, 0] = cx + (polygon[:, 0] - cx) * shrink_ratio
        polygon[:, 1] = cy + (polygon[:, 1] - cy) * shrink_ratio
        return polygon

    @staticmethod
    def shrink_polygon_pyclipper(polygon: np.ndarray, shrink_ratio: float) -> np.ndarray:
        """


        Args:
            polygon:
            shrink_ratio:

        Returns:

        """
        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked = padding.Execute(-distance)
        if not shrinked:
            shrinked = np.array(shrinked)
        else:
            shrinked = np.array(shrinked[0]).reshape(-1, 2)
        return shrinked
