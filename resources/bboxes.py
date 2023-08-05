from typing import List, Union, Any

import cv2
import numpy as np


def get_rect(item: Union[np.ndarray, Any]) -> Any:
    """
    Auxiliary function which calculates a bounding box for an incoming object.
    Args:
        item: np.ndarray of points or an object that has "bbox" attribute

    Returns:
        (x, y, width, height) of the outcoming bounding box
    """
    if isinstance(item, np.ndarray):
        return cv2.boundingRect(item)
    else:
        if hasattr(item, "bbox"):
            return cv2.boundingRect(item.bbox.astype(np.float32))
        else:
            raise NotImplementedError("An incoming object should has either type np.ndarray, or an attribute 'bbox'.")


def sort_boxes(items: List[Union[np.ndarray, Any]], sorting_type: str = 'top2down') -> List[Any]:
    """
    Auxiliary function for sorting objects in order from up to down and from left to right.
    Args:
        items: objects for sorting
        sorting_type: top2down or left2right

    Returns:
        sorted objects
    """
    bboxes = [get_rect(item) for item in items]

    # NOTE: cv2.boundingRect -> x, y, w, h
    if sorting_type == 'top2down':
        sort_lambda = lambda pair: pair[0][1]
    elif sorting_type == 'left2right':
        sort_lambda = lambda pair: pair[0][0]
    else:
        raise NotImplementedError(f'Sorting method "{sorting_type}" is not supported.')
    items = [x for _, x in sorted(zip(bboxes, items), key=sort_lambda)]
    return items


def sort_boxes_top2down_wrt_left2right_order(items: List[Union[np.ndarray, Any]]) -> List[Any]:
    """
    Auxiliary function for sorting objects following their order on the page. Algorithm:
    1. Sorting all words top-down.
    2. Choose a word on the top that doesn't have another word, whose middle is upper than a lower edge of the initial word.
    3. Add the word to a new list.

    Args:
        items: objects for sorting
    Returns:
        sorted objects
    """
    if len(items) == 0:
        return []

    if len(items) == 1:
        return items

    def is_leftmost(item: Union[np.ndarray, Any], line_items: List[Union[np.ndarray, Any]]) -> bool:
        """
        Auxiliary function that calculates whether the object is the most left, and doesn't have other objects whose middle is upper than a lower edge of the object,

        Args:
            item: objects for checking
            line_items: list of objects nearby

        Returns:
            True or False
        """
        item_rect = get_rect(item)
        baseline_y = item_rect[1] + item_rect[3]

        for line_item in line_items:
            line_item_rect = get_rect(line_item)
            # Take more left objects
            if line_item_rect[0] < item_rect[0]:
                middle_point_y = line_item_rect[1] + line_item_rect[3] // 2

                # If the middle line is upper than the base line than looking for a next object
                if middle_point_y < baseline_y:
                    return False
        return True

    items = sort_boxes(items, sorting_type='top2down')

    out_items = []
    while len(items) > 0:
        for idx, item in enumerate(items):
            if is_leftmost(item, items):
                out_items.append(items.pop(idx))
                break

    return out_items


def fit_bbox(bboxes: np.ndarray) -> np.ndarray:
    left = bboxes[:, :, 0].reshape(-1).min()
    right = bboxes[:, :, 0].reshape(-1).max()
    top = bboxes[:, :, 1].reshape(-1).min()
    bottom = bboxes[:, :, 1].reshape(-1).max()

    bbox = np.array([
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom]
    ])
    return bbox
