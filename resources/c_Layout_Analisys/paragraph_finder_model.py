import math

from sklearn.cluster import DBSCAN

import numpy as np
from typing import List
import sys
#sys.path.append(r'.\resources\e_Service_Deployment') # import utils
sys.path.append(r'.\resources')  # from a_Text_Detection.utils import
#sys.path.append(r'.\resources\a_Text_Detection')  # from IntelligentDocumentProcessing.Resources.a_Text_Detection.utils import
sys.path.append('./')

from c_Layout_Analisys.utils import sort_boxes_top2down_wrt_left2right_order, sort_boxes, fit_bbox, Paragraph, Line


def angle_of_vectors(a, b, c, d):
    dotProduct = a @ c[np.newaxis].T + b @ d[np.newaxis].T
    print(dotProduct)
    # for three dimensional simply add dotProduct = a*c + b*d  + e*f
    modOfVector1 = math.sqrt(a @ a[np.newaxis].T + b @ b[np.newaxis].T) * math.sqrt(
        c @ c[np.newaxis].T + d @ d[np.newaxis].T)
    # for three dimensional simply add modOfVector = math.sqrt( a*a + b*b + e*e)*math.sqrt(c*c + d*d +f*f)
    angle = dotProduct / modOfVector1
    print("Cosθ =", angle)
    angleInDegree = math.degrees(math.acos(angle))
    print("θ =", angleInDegree, "°")


# КОД ДЛЯ СТУДЕНТА
class ParagraphFinder:
    def __init__(
            self,
            angle_threshold: int = 10,
            overlapping_threshold: float = 0.3,
            max_distance: int = 1,
            cluster_eps: float = 0.5
    ):
        self.angle_threshold = angle_threshold
        self.overlapping_threshold = overlapping_threshold
        self.max_distance = max_distance
        self.cluster_eps = cluster_eps

    @staticmethod
    def _distance(line1: Line, line2: Line) -> float:
        # return np.linalg.norm(np.array(
        #    line1.normalized_bbox[:, 1].mean()
        #    - line2.normalized_bbox[:, 1].mean()
        # ))
        (x11, y11), (x12, y12), _, _ = line1.normalized_bbox
        (x21, y21), (x22, y22), _, _ = line2.normalized_bbox
        y1m = (y12 + y11) / 2
        y2m = (y22 + y21) / 2
        return abs(y1m - y2m)

    @staticmethod
    def _length(line: Line) -> float:
        return np.linalg.norm(line.normalized_bbox[0] - line.normalized_bbox[1])

    def _overlapping(self, line1: Line, line2: Line) -> float:
        # return abs(
        #    min(line1.normalized_bbox[1, 0], line2.normalized_bbox[1, 0])
        #    - max(line1.normalized_bbox[0, 0], line2.normalized_bbox[0, 0])
        # ) / self._length(line1)
        (x11, y11), (x12, y12), _, _ = line1.normalized_bbox
        (x21, y21), (x22, y22), _, _ = line2.normalized_bbox
        x_min = max(x11, x21)
        x_max = min(x12, x22)

        if x_max - x_min < 0:
            return 0

        return (x_max - x_min) / self._length(line1)

    @staticmethod
    def _angle(line1: Line, line2: Line) -> float:
        (x11, y11), (x12, y12), _, _ = line1.normalized_bbox
        (x21, y21), (x22, y22), _, _ = line2.normalized_bbox
        d_x1 = x12 - x11
        d_y1 = y12 - y11
        d_x2 = x22 - x21
        d_y2 = y22 - y21

        theta = math.atan(d_y2 / d_x2) - math.atan(d_y1 / d_x1)
        theta = math.degrees(theta)

        if theta > 90:
            theta -= 180
        if theta < -90:
            theta += 180

        return theta

    def paragraph_distance(self, line1: np.ndarray, line2: np.ndarray) -> float:
        line1 = Line(normalized_bbox=line1.reshape(4, 2))
        line2 = Line(normalized_bbox=line2.reshape(4, 2))

        if abs(self._angle(line1, line2)) > self.angle_threshold:
            return self.max_distance

        if self._overlapping(line1, line2) < self.overlapping_threshold:
            return self.max_distance

        return self._distance(line1, line2)

    @staticmethod
    def prepare_lines(lines: List[Line]) -> np.ndarray:
        return np.array([line.normalized_bbox.reshape(-1) for line in lines])

    def fit_cluster(self, lines: List[Line]) -> DBSCAN:
        prepared_lines = self.prepare_lines(lines)
        cluster = DBSCAN(metric=self.paragraph_distance, eps=self.cluster_eps)
        cluster.fit(prepared_lines)
        return cluster

    @staticmethod
    def sort_paragraphs(paragraphs):
        for par in paragraphs:
            par.items = sort_boxes(par.items, sorting_type='top2down')  # сортировка сверху вниз
        paragraphs = sort_boxes_top2down_wrt_left2right_order(paragraphs)  # сортировка в порядке чтения
        return paragraphs

    def find_paragraphs(self, lines: List[Line]) -> List[Paragraph]:
        cluster = self.fit_cluster(lines)

        paragraphs = []
        for label in set(cluster.labels_):
            line_indexes = np.argwhere(cluster.labels_ == label)
            par_lines = [lines[idx[0]] for idx in line_indexes]
            bbox = fit_bbox(np.array([lines[idx[0]].bbox for idx in line_indexes]))
            par = Paragraph(items=par_lines, bbox=bbox)
            paragraphs.append(par)
        paragraphs = self.sort_paragraphs(paragraphs)
        return paragraphs