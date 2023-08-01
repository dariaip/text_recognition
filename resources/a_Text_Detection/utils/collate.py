from typing import Any, List

import torch


def collate(batch: Any) -> List[Any]:
    """
    Вспомогательная функция для объединения всех элементов батча в тензоры для обучения.

    Args:
        batch: набор данных для обучения из датасета

    Returns:
        подготовленный набор данных для обучения
    """
    out_raw_imgs = []
    out_imgs = []
    out_shrink_maps = []
    out_shrink_masks = []
    out_threshold_maps = []
    out_threshold_masks = []
    out_gt_polygons = []

    for raw_img, img, s_map, s_mask, t_map, t_mask, p in batch:
        out_raw_imgs.append(raw_img)
        out_imgs.append(img)
        out_shrink_maps.append(s_map if isinstance(s_map, torch.Tensor) else torch.tensor(s_map))
        out_shrink_masks.append(s_mask if isinstance(s_mask, torch.Tensor) else torch.tensor(s_mask))
        out_threshold_maps.append(t_map if isinstance(t_map, torch.Tensor) else torch.tensor(t_map))
        out_threshold_masks.append(t_mask if isinstance(t_mask, torch.Tensor) else torch.tensor(t_mask))
        out_gt_polygons.append(p)

    return [
        out_raw_imgs,
        torch.stack(out_imgs),
        torch.stack(out_shrink_maps),
        torch.stack(out_shrink_masks),
        torch.stack(out_threshold_maps),
        torch.stack(out_threshold_masks),
        out_gt_polygons
    ]
