from typing import List, Any, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class PadCollate(object):
    def __init__(self, pad_value: float = 0.):
        """
        Класс-прокси для объединения данных в один батч и выравнивания всех изображений по ширине.

        Args:
            pad_value: значение, которым будут паддиться тензоры с изображениями
        """
        self.pad_value = pad_value

    def __call__(self, batch: List[Any]) -> Tuple[torch.Tensor, List[Any], List[str], List[Any], torch.Tensor]:
        """
        Метод, который принимает на вход батч и приводит все изображения внутри к одной длине.

        Args:
            batch: входной батч

        Returns:
            развернутый батч, готовый для подачи в модель
        """
        images, raw_images, labels, targets, lengths = zip(*batch)
        max_width = max([image.shape[2] for image in images])

        targets = pad_sequence(targets).squeeze(2).permute(1, 0)
        lengths = torch.IntTensor(lengths)

        padded_images = []
        for image in images:
            c, h, w = image.shape
            image = F.pad(image, (0, max_width - w), value=self.pad_value)
            padded_images.append(image)

        image_tensors = torch.stack(padded_images, dim=0)
        return image_tensors, raw_images, labels, targets, lengths
