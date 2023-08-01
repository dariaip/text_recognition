from typing import Union, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BalanceCrossEntropyLoss(nn.Module):
    def __init__(self, negative_ratio: float = 3.0, eps: float = 1e-6):
        """
        Balanced cross entropy loss с учетом негативных примеров (easy negative mining).

        Args:
            negative_ratio: отношение числа негативных примеров к числу позитивных
            eps: константа для избежания деления на 0
        """
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
        return_origin: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Args:
            pred: предсказания модели, размерность (N, H, W)
            gt: истинные значения, размерность (N, H, W),
            mask: маска с положительными регионами, размерность (N, H, W)
            return_origin: возвращать ли значение binary_cross_entropy
        """
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))
        loss = F.binary_cross_entropy(pred, gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = negative_loss.view(-1).topk(negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)

        if return_origin:
            return balance_loss, loss
        return balance_loss


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        """
        Dice loss из статьи https://arxiv.org/abs/1707.03237.

        Args:
            eps: константа для избежания деления на 0
        """
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: предсказания модели, размерность (N, H, W) или (N, 1, H, W)
            gt: истинные значения, размерность (N, H, W) или (N, 1, H, W)
            mask: маска с положительными регионами, размерность (N, H, W)
            weights: (опционально) веса, которые умножаются на маску
        """
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
            gt = gt[:, 0, :, :]
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        intersection = (pred * gt * mask).sum()

        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class MaskL1Loss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        """
        L1 loss с учетом маски.

        Args:
            eps: константа для избежания деления на 0
        """
        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: предсказания модели, размерность (N, H, W)
            gt: истинные значения, размерность (N, H, W),
            mask: маска с положительными регионами, размерность (N, H, W)
        """
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        return loss


class DBLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 10.0,
        ohem_ratio: float = 3.0,
        reduction: str = 'mean',
        eps: float = 1e-6
    ):
        """
        Кастомный лосс, включающий в себя Cross Entropy loss, Dice loss и L1 loss.

        Args:
            alpha: коэффициент для Cross Entropy loss
            beta: коэффициент для L1 loss
            ohem_ratio: отношение числа негативных примеров к числу позитивных в Cross Entropy loss
            reduction: способ агрегации лосса, может быть mean (усреднение) или sum (суммирование)
            eps: константа для избежания деления на 0
        """
        super().__init__()
        assert reduction in ['mean', 'sum'], "параметр reduction должен быть одним из ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio)
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        shrink_maps = pred[:, 0, :, :]
        threshold_maps = pred[:, 1, :, :]

        loss_shrink_maps = self.bce_loss(shrink_maps, batch['shrink_map'], batch['shrink_mask'])
        loss_threshold_maps = self.l1_loss(threshold_maps, batch['threshold_map'], batch['threshold_mask'])

        metrics = dict(
            loss_shrink_maps=loss_shrink_maps,
            loss_threshold_maps=loss_threshold_maps,
        )
        if pred.size()[1] > 2:
            binary_maps = pred[:, 2, :, :]
            loss_binary_maps = self.dice_loss(binary_maps, batch['shrink_map'], batch['shrink_mask'])
            metrics['loss_binary_maps'] = loss_binary_maps
            loss_all = self.alpha * loss_shrink_maps + self.beta * loss_threshold_maps + loss_binary_maps
            metrics['loss'] = loss_all
        else:
            metrics['loss'] = loss_shrink_maps
        return metrics
