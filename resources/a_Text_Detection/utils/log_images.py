from typing import List, Dict

import numpy as np
from pytorch_lightning.loggers import WandbLogger

from .drawmore import DrawMore


def log_images(
    logger,
    raw_images: List[np.ndarray],
    shrink_maps: np.ndarray,
    threshold_maps: np.ndarray,
    pred_bin_maps: np.ndarray,
    pred_shrink_maps: np.ndarray,
    pred_threshold_maps: np.ndarray,
    gt_polygons: List[List[np.ndarray]],
    polygons_after_step: List[List[np.ndarray]],
    metrics_after_step: Dict,
    mode: str,
    log_max: int = 10
) -> None:
    if not isinstance(logger, WandbLogger):
        return

    logger.experiment.log(
        {f"{mode}_gt/original": DrawMore.to_wandb([DrawMore.make_grid(raw_images[:log_max])])}
    )
    logger.experiment.log(
        {f"{mode}_gt/shrink_maps": DrawMore.to_wandb([DrawMore.make_grid(DrawMore.to_cv2(shrink_maps[:log_max]))])}
    )
    logger.experiment.log(
        {f"{mode}_gt/threshold_maps": DrawMore.to_wandb(
            [DrawMore.make_grid(DrawMore.to_cv2(threshold_maps[:log_max]))])}
    )
    logger.experiment.log(
        {f"{mode}_preds/pred_bin_maps": DrawMore.to_wandb(
            [DrawMore.make_grid(DrawMore.to_cv2(pred_bin_maps[:log_max]))])}
    )
    logger.experiment.log(
        {f"{mode}_preds/pred_shrink_maps": DrawMore.to_wandb(
            [DrawMore.make_grid(DrawMore.to_cv2(pred_shrink_maps[:log_max]))])}
    )
    logger.experiment.log(
        {f"{mode}_preds/pred_threshold_maps": DrawMore.to_wandb(
            [DrawMore.make_grid(DrawMore.to_cv2(pred_threshold_maps[:log_max]))])}
    )

    pred_image = DrawMore.draw_match_result(
        raw_images[0],
        gt_polygons[0],
        polygons_after_step[0],
        metrics_after_step['iou_thresholds'],
        metrics_after_step['matched_indices'][0]
    )
    logger.experiment.log({f"{mode}_preds/preds": DrawMore.to_wandb([pred_image])})
