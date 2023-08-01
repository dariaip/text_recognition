from typing import List, Union, Any

import cv2
import numpy as np


def get_rect(item: Union[np.ndarray, Any]) -> Any:
    """
    Вспомогательная функция, вычисляюшая обрамляющий прямоугольник входного объекта.
    Args:
        item: массив точек типа np.ndarray или объект, имеющий атрибут bbox

    Returns:
        x, y, ширина, высота обрамляющего прямоугольника
    """
    if isinstance(item, np.ndarray):
        return cv2.boundingRect(item)
    else:
        if hasattr(item, "bbox"):
            return cv2.boundingRect(item.bbox.astype(np.float32))
        else:
            raise NotImplementedError("Входной объект должен быть типа np.ndarray или иметь атрибут bbox.")


def sort_boxes(items: List[Union[np.ndarray, Any]], sorting_type: str = 'top2down') -> List[Any]:
    """
    Вспомогательная функция сортировки объектов сверху вниз или слева направо.
    Args:
        items: объекты для сортировки
        sorting_type: top2down - сверху вниз; left2right - слева направо

    Returns:
        отсортированные объекты
    """
    bboxes = [get_rect(item) for item in items]

    # NOTE: cv2.boundingRect -> x, y, w, h
    if sorting_type == 'top2down':
        sort_lambda = lambda pair: pair[0][1]
    elif sorting_type == 'left2right':
        sort_lambda = lambda pair: pair[0][0]
    else:
        raise NotImplementedError(f'Метод сортировки "{sorting_type}" не поддерживается.')
    items = [x for _, x in sorted(zip(bboxes, items), key=sort_lambda)]
    return items


def sort_boxes_top2down_wrt_left2right_order(items: List[Union[np.ndarray, Any]]) -> List[Any]:
    """
    Вспомогательная функция сортировки объектов с учетом порядка следования на странице. Алгоритм:
    1. Сортируем все слова сверху вниз.
    2. Выбираем самое верхнее слово, у которого слева нет другого слова, середина которого выше нижней границы
    текущего слова.
    3. Добавляем его в новый список.

    Args:
        items: список объектов для сортировки
    Returns:
        отсортированные с учетом порядка следования на странице объекты
    """
    if len(items) == 0:
        return []

    if len(items) == 1:
        return items

    def is_leftmost(item: Union[np.ndarray, Any], line_items: List[Union[np.ndarray, Any]]) -> bool:
        """
        Вспомогательная функция, вычисляющая, является ли объект самым левым, не имеющим слева других объектов,
        середина которых выше нижней границы данного.

        Args:
            item: объект для проверки
            line_items: список объектов, соседних с данным

        Returns:
            True, если объект является самым левым, не имеющим слева других объектов, середина которых выше нижней
            границы данного; иначе False
        """
        item_rect = get_rect(item)
        baseline_y = item_rect[1] + item_rect[3]

        for line_item in line_items:
            line_item_rect = get_rect(line_item)
            # Берем элементы левее текущего
            if line_item_rect[0] < item_rect[0]:
                middle_point_y = line_item_rect[1] + line_item_rect[3] // 2

                # Если средняя линия выше базовой линии то ищем другой элемент
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
