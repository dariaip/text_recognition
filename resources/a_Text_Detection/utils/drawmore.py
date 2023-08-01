from typing import List, Tuple

import cv2
import numpy as np
import torch
import wandb


class DrawMore:
    """
    Вспомогательный класс с набором методов для конвертации и отрисовки.
    """

    @staticmethod
    def draw_match_result(
        image: np.ndarray,
        gt_polygons: List[np.ndarray],
        pred_polygons: List[np.ndarray],
        matched_ious: np.ndarray,
        matched_indices: np.ndarray
    ) -> np.ndarray:
        """
        Вспомогательный метод для отрисовки предсказанных полигонов и истинных полигонов на изображении.

        Args:
            image: изображение
            gt_polygons: истинные полигоны с текстом
            pred_polygons: предсказанные полигоны с текстом
            matched_ious: IOU у совпавших полигонов
            matched_indices: индексы совпавших полигонов

        Returns:
            изображение с отрисованными истинными и предсказанными полигонами
        """
        gt = DrawMore.draw_contours(image, gt_polygons, color=(0, 0, 255))

        iou_index = np.argwhere(matched_ious.astype(np.float32) == 0.5).min()
        matched_indices = np.array(matched_indices[iou_index])

        gt_matched_indices = set(matched_indices[:, 0]) if len(
            matched_indices) else []
        pred_matched_indices = set(matched_indices[:, 1]) if len(
            matched_indices) else []

        tp_polygons = [p for idx, p in enumerate(
            pred_polygons) if idx in pred_matched_indices]
        fp_polygons = [p for idx, p in enumerate(
            pred_polygons) if idx not in pred_matched_indices]
        fn_polygons = [p for idx, p in enumerate(
            gt_polygons) if idx not in gt_matched_indices]

        preds = DrawMore.draw_contours(image, tp_polygons, color=(0, 255, 0))
        preds = DrawMore.draw_contours(preds, fp_polygons, color=(255, 0, 0), inplace=True)
        preds = DrawMore.draw_contours(preds, fn_polygons, color=(0, 0, 255), inplace=True)

        result = np.concatenate([gt, preds], axis=1)
        return result

    @staticmethod
    def draw_contours(
        image: np.ndarray,
        polygons: List[np.ndarray],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        fill: bool = False,
        inplace: bool = False
    ) -> np.ndarray:
        """
        Вспомогательный метод для отрисовки набора контуров (полигонов) на изображении.

        Args:
            image: изображение
            polygons: список полигонов
            color: цвет отрисовки полигонов
            thickness: толщина отрисовки полигонов
            fill: заливать ли отрисованные полигоны цветом
            inplace: изменять ли исходное изображение

        Returns:
            исходное изображение, если inplace=True, иначе копия исходного изображения с отрисованными полигонами
        """
        if not inplace:
            image = image.copy()

        for p in polygons:
            p = np.array([p]).astype(np.int32)
            if fill:
                cv2.fillPoly(image, p, color)
            else:
                cv2.drawContours(image, p, -1, color, thickness=thickness)
        return image

    @staticmethod
    def to_numpy(tensor: torch.Tensor, unsqueeze: bool = False, apply_sigmoid: bool = False) -> np.ndarray:
        """
        Вспомогательный метод для превращения тензора в np.ndarray.

        Args:
            tensor: тензор
            unsqueeze: добавлять ли 1 размерность к тензору
            apply_sigmoid: применять ли сигмоиду к тензору

        Returns:
            преобразованный тензор в виде np.ndarray
        """
        if unsqueeze:
            tensor = tensor.unsqueeze(1)
        if apply_sigmoid:
            tensor = tensor.sigmoid()
        return tensor.detach().cpu().numpy()

    @staticmethod
    def make_grid(images: List[np.ndarray]) -> np.ndarray:
        """
        Вспомогательный метод, объединяющий список изображений типа np.ndarray в один массив.

        Args:
            images: список изображений типа np.ndarray

        Returns:
            объединенный массив со всеми изображениями
        """
        return np.concatenate(images, axis=1)

    @staticmethod
    def to_cv2(images: np.ndarray, unsqueeze: bool = False) -> List[np.ndarray]:
        """
        Вспомогательный метод для превращения списка нормализованных массивов np.ndarray в список изображений.

        Args:
            images: список нормализованных массивов типа np.ndarray
            unsqueeze: добавлять ли 0 размерность к массиву

        Returns:
            список изображений
        """
        if unsqueeze:
            images = [np.expand_dims(image, 0) for image in images]
        return [np.moveaxis(image * 255, 0, 2).astype(np.uint8) for image in images]

    @staticmethod
    def to_wandb(images: List[np.ndarray]) -> List[wandb.Image]:
        """
        Вспомогательный метод для превращения списка изображений в изображения типа.wandb.Image.

        Args:
            images: список изображений

        Returns:
            список изображений типа.wandb.Image
        """
        return [wandb.Image(image) for image in images]
