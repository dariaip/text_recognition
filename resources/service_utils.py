import math
from typing import Any, List, Union

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from resources.points2polygon import points2polygon
from resources.dtos import Word, Line, Paragraph


def four_point_transform(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Auxiliary function for cutting a crop from the provided image according to the provided coordinates of a bounding box.
    Source: https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    Args:
        image: an image
        box: coordinates of a bounding box in format [[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]]

    Returns:
        the crop
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
    Auxiliary function that cuts crops from the provided image according to all provided bounding boxes.

    Args:
        image: an image
        bboxes: list of coordinates of bounding boxes

    Returns:
        the crops
    """
    crops = []
    for box in bboxes:
        crop = four_point_transform(image, box)
        crops.append(crop)
    return crops


def batchings(objects: List[Any], batch_size: int) -> List[Any]:
    """
    Auxiliary function (generator) for getting consecutive sets of the provided objects where each set includes batch_size elements.
    For instance, batchings(list(range(10), 4)) returns [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]].

    Args:
        objects: a list of objects
        batch_size: size of selected sets

    Returns:
        The function is a generator that's why each iteration leads to returning a following set
    """
    for i in range(0, len(objects), batch_size):
        yield objects[i: i + batch_size]


def get_iou(p1: Polygon, p2: Polygon) -> float:
    """
    Auxiliary function that calculates a customized Intersection over Union metrics for two provided polygons.

    Args:
        p1: the 1st polygon
        p2: the 2nd polygon

    Returns:
        a ratio between an area of intersection of the provided polygons to area of the first polygon
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
    Auxiliary function for joining a list of words (Word) and a list of lines' coordinates
    or joining a list of lines (Line) and a list of groups' coordinates.

    Args:
        inner_elements: list of words (each word of type "Word") list of lines (type "Line")
        outer_element_bboxes: list of coordinates of lines or groups

    Returns:
        list of lines (type "Line") or list of groups (type "Group")
    """
    outer_element_type = Paragraph if isinstance(inner_elements[0], Line) else Line

    inner_element_polygons = [points2polygon(el.bbox) for el in inner_elements]
    outer_element_polygons = [points2polygon(el.bbox) for el in outer_elements]
    intersection_matrix = np.full((len(inner_elements), len(outer_elements)), fill_value=0, dtype=float)
    s = STRtree(inner_element_polygons)
    for pred_idx, pred_polygon in enumerate(outer_element_polygons):
        result = s.query(pred_polygon)
        if len(result) > 0:
            for gt_idx in result:
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
