from typing import Optional, Union, List, Dict, Any, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from .points2polygon import points2polygon


class BaseEvaluator:
    """
    Класс для подсчета метрик Precision, Recall F-score.
    """
    def __call__(self, gt_polygons: List[List[np.ndarray]], pred_polygons: List[List[np.ndarray]]) -> Dict[str, Any]:
        """
        Основной метод для подсчета метрик Precision, Recall F-score.

        Args:
            gt_polygons: истинные полигоны одного примера
            pred_polygons: предсказанные полигоны одного примера

        Returns:
            словарь с усредненными метриками mean_precision, mean_recall, mean_fscore, support, всеми метриками
                precisions, recalls, fscores, а также support, iou_thresholds и matched_indices (индексы совпавших
                ббоксов)
        """
        out = [self.mean_precision_recall_fscore_support(gt_p, pred_p)
               for gt_p, pred_p in zip(gt_polygons, pred_polygons)]

        result = {
            'mean_precision': np.mean([o['mean_precision'] for o in out]),
            'mean_recall': np.mean([o['mean_recall'] for o in out]),
            'mean_fscore': np.mean([o['mean_fscore'] for o in out]),
            'mean_support': np.mean([o['support'] for o in out]),
            'iou_thresholds': out[0]['iou_thresholds'],
            'precisions': np.mean(np.stack([o['precisions'] for o in out]), axis=0),
            'recalls': np.mean(np.stack([o['recalls'] for o in out]), axis=0),
            'fscores': np.mean(np.stack([o['fscores'] for o in out]), axis=0),
            'matched_indices': [o['matched_indices'] for o in out]
        }
        return result

    def mean_precision_recall_fscore_support(
        self,
        gt_points: List[np.ndarray],
        pred_points: List[np.ndarray],
        iou_thresholds: Optional[Union[List[float], np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Метод для подсчета метрик Precision, Recall, F-score.

        Args:
            gt_points: истинные полигоны одного примера
            pred_points: предсказанные полигоны одного примера
            iou_thresholds: отсечки для подсчета метрики по IOU, по умолчанию [0.3, 0.95, 0.05]

        Returns:
            словарь с усредненными метриками mean_precision, mean_recall, mean_fscore, support, всеми метриками
                precisions, recalls, fscores, а также support, iou_thresholds и matched_indices (индексы совпавших ббоксов)
        """
        if iou_thresholds is None:
            iou_thresholds = np.arange(0.3, 0.95, 0.05)
        gt_polygons = [points2polygon(p, idx) for idx, p in enumerate(gt_points)]
        pred_polygons = [points2polygon(p, idx) for idx, p in enumerate(pred_points)]
        s = STRtree(gt_polygons)
        iou_matrix = np.full((len(gt_polygons), len(pred_polygons)), fill_value=0, dtype=float)

        for pred_idx, pred_polygon in enumerate(pred_polygons):
            result = s.query(pred_polygon)
            if len(result) > 0:
                for gt_polygon in result:
                    gt_idx = gt_polygon.idx

                    iou_matrix[gt_idx, pred_idx] = self.get_iou(pred_polygon, gt_polygon)

        ps, rs, fs, ss, ts, midxs = [], [], [], [], [], []
        for t in iou_thresholds:
            t = round(t, 2)
            p, r, f, s, matched_indices = self.precision_recall_fscore_support(iou_matrix, iou_threshold=t)
            ps.append(p)
            rs.append(r)
            fs.append(f)
            ss.append(s)
            ts.append(t)
            midxs.append(matched_indices)

        return {
            'mean_precision': np.mean(ps),
            'mean_recall': np.mean(rs),
            'mean_fscore': np.mean(fs),
            'support': np.mean(ss),
            'iou_thresholds': iou_thresholds,
            'precisions': ps,
            'recalls': rs,
            'fscores': fs,
            'matched_indices': midxs
        }

    @staticmethod
    def get_iou(p1: Polygon, p2: Polygon) -> float:
        """
        Вспомогательный метод, подсчитывающий intersection over union (IOU) для двух полигонов.

        Args:
            p1: первый полигон
            p2: второй полигон

        Returns:
            IOU для двух входных полигонов
        """
        intersection = p1.intersection(p2).area
        union = p1.area + p2.area
        return intersection / (union - intersection)

    @staticmethod
    def linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
        """
        Вспомогательный метод, решающий задачу о назначениях для заданной матрицы.

        Args:
            cost_matrix: матрица

        Returns:
            индексы строк и соответствующих столбцов оптимального решения задачи о назначениях в виде списка кортежей
        """
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

    def associate_gt_to_preds(self, iou_matrix: np.ndarray, iou_threshold: float = 0.5) -> Tuple[int, int, int, List[int]]:
        """
        Вспомогательный метод, ассоциирующий индексы предсказаний и истинных значений по IOU. Принимает на вход
        матрицу размером (количество истинных ббоксов; количество предсказанных ббоксов),
        в ячейке (i, j) которой находится значение IOU для i-го истинного ббокса и j-го предсказанного, и по ней вычисляет
        наиболее пересекающиеся истинные и предсказанные ббоксы, IOU которых больше заданного iou_threshold.

        Args:
            iou_matrix: матрица со значениями IOU для истинных и предсказанных ббоксов
            iou_threshold: граница IOU, выше которой ббоксы считаются совпавшими

        Returns:
            количество true positive, false positive, false negative, а также индексы совпавших ббоксов
        """
        gt_num, pred_num = iou_matrix.shape
        matched_indices = self.linear_assignment(-iou_matrix)
        matched_ious = np.array([iou_matrix[i, j] for i, j in matched_indices])
        matched_indices = [idx for idx, iou in zip(matched_indices, matched_ious) if iou > iou_threshold]
        tp = len(matched_indices)
        fp = pred_num - tp
        fn = gt_num - tp
        return tp, fp, fn, matched_indices

    def precision_recall_fscore_support(
        self,
        iou_matrix: np.ndarray,
        iou_threshold: float = 0.5
    ) -> Tuple[float, float, float, int, List[int]]:
        """
        Вспомогательный метод, принимающая на вход матрицу IOU и считающая метрики Precision, Recall, F-score, support
        и вычисляющая индексы совпавших ббоксов с помощью метода associate_gt_to_preds.

        Args:
            iou_matrix: матрица со значениями IOU для истинных и предсказанных ббоксов
            iou_threshold: граница IOU, выше которой ббоксы считаются совпавшими

        Returns:
            значения метрик Precision, Recall, F-score, support и индексы совпавших ббоксов
        """
        tp, fp, fn, matched_indices = self.associate_gt_to_preds(iou_matrix, iou_threshold)

        precision = tp / (tp + fp) if tp + fp > 0 else 0.
        recall = tp / (tp + fn) if tp + fn > 0 else 0.
        fscore = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.
        support = tp + fp + fn

        return precision, recall, fscore, support, matched_indices
