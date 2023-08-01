import math
from typing import Any, List, Union, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from resources.drawmore import DrawMore
from resources.points2polygon import points2polygon
from resources.dtos import Word, Line, Paragraph


def four_point_transform(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Вспомогательная функция для вырезания кропа из исходного изображения по координатам 4 точек обрамляющего
    прямоугольника.
    Оригинал: https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    Args:
        image: исходное изображение
        box: координаты обрамляющего прямоугольника в виде [[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]]

    Returns:
        кроп, вырезанный по координатам обрамляющего прямоугольника
    """
    (tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y) = box.tolist()
    width_a = (tr_x - tl_x) ** 2 + (tr_y - tl_y) ** 2
    width_b = (br_x - bl_x) ** 2 + (br_y - bl_y) ** 2
    max_width = round(math.sqrt(max(width_a, width_b)))
    height_a = (br_x - tr_x) ** 2 + (br_y - tr_y) ** 2
    height_b = (bl_x - tl_x) ** 2 + (bl_y - tl_y) ** 2
    max_height = round(math.sqrt(max(height_a, height_b)))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(box.astype(np.float32), dst)
    warped = cv2.warpPerspective(
        image, matrix, (max_width, max_height), flags=cv2.INTER_LINEAR
    )

    return warped


def prepare_crops(image: np.ndarray, bboxes: List[np.ndarray]) -> List[np.ndarray]:
    """
    Вспомогательная функция, которая вырезает кропы из исходного изображения по всем обрамляющим прямоугольникам.

    Args:
        image: исходное изображение
        bboxes: список из координат обрамляющих прямоугольников

    Returns:
        кропы по всем обрамляющим прямоугольникам
    """
    crops = []
    for box in bboxes:
        crop = four_point_transform(image, box)
        crops.append(crop)
    return crops


def batchings(objects: List[Any], batch_size: int) -> List[Any]:
    """
    Вспомогательная функция-генератор, возвращающая последовательные подмножества входного списка размером batch_size.
    Например, batchings(list(range(10), 4)) при приведении к списку вернет [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]].

    Args:
        objects: список объектов
        batch_size: размер целевых подмножеств

    Returns:
        так как данная функция - это генератор, то на каждой итерации она вернет очередное подмножество исходного списка
    """
    for i in range(0, len(objects), batch_size):
        yield objects[i: i + batch_size]


def get_iou(p1: Polygon, p2: Polygon) -> float:
    """
    Вспомогательная функция, вычисляющая отношение площади пересечения двух полигонов к площади первого полигона.

    Args:
        p1: первый полигон
        p2: второй полигон

    Returns:
        отношение площади пересечения двух полигонов к площади первого полигона
    """
    intersection = p1.intersection(p2).area
    return intersection / p1.area


def group_words_by_lines_or_lines_by_paragraphs(
    inner_elements: Union[List[Word], List[Line]],
    outer_elements: Union[List[Line], List[Paragraph]],
    w: int,
    h: int
) -> Union[List[Line], List[Paragraph]]:
    """
    Вспомогательная функция, объединяющая список слов (Word) и список координат линий или
    список линий (Line) и список координат групп.

    Args:
        inner_elements: список слов типа Word или линий типа Line
        outer_element_bboxes: список координат линий или групп

    Returns:
        список линий типа Line или групп типа Group
    """
    outer_element_type = Paragraph if isinstance(inner_elements[0], Line) else Line

    inner_element_polygons = [points2polygon(el.bbox, idx) for idx, el in enumerate(inner_elements)]
    outer_element_polygons = [points2polygon(el.bbox, idx) for idx, el in enumerate(outer_elements)]
    intersection_matrix = np.full((len(inner_elements), len(outer_elements)), fill_value=0, dtype=float)
    s = STRtree(inner_element_polygons)
    for pred_idx, pred_polygon in enumerate(outer_element_polygons):
        result = s.query(pred_polygon)
        if len(result) > 0:
            for gt_idx in result:
                #gt_idx = gt_polygon.idx
                gt_polygon = inner_element_polygons[gt_idx]
                intersection_matrix[gt_idx, pred_idx] = get_iou(gt_polygon, pred_polygon)
    inner_with_no_outer = [idx for idx, max_iou in enumerate(np.max(intersection_matrix, axis=1))
                           if max_iou == 0]
    inner_outer_indexes = np.argmax(intersection_matrix, axis=1)

    new_outer_elements = []
    for inner_idx, outer_idx in enumerate(inner_outer_indexes):
        inner_element = inner_elements[inner_idx]
        if inner_idx in inner_with_no_outer:
            bbox = inner_element.bbox
            kwargs = {'bbox': bbox, 'items': [inner_element]}
            if outer_element_type == Line:
                norm_bbox = bbox.copy()
                norm_bbox[:, 0] /= w
                norm_bbox[:, 1] /= h
                kwargs.update({'normalized_bbox': norm_bbox})
            new_outer_elements.append(outer_element_type(**kwargs))
        else:
            outer_elements[outer_idx].items.append(inner_element)
    new_outer_elements = outer_elements + new_outer_elements

    return new_outer_elements
