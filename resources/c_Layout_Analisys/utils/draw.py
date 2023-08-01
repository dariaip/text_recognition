import random
from typing import List

import cv2
import numpy as np
from matplotlib import pyplot as plt

from resources.a_Text_Detection.utils import DrawMore
from .dtos import Paragraph


def get_random_color() -> List[int]:
    rnd_hue = random.randint(0, 359)
    color_hsv = np.array([[[rnd_hue, 255, 255]]], dtype=np.uint8)
    color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)
    return color_rgb[0, 0, :].tolist()


def draw_paragraphs(image: np.ndarray, paragraphs: List[Paragraph]) -> None:
    paragraph_result_image = image.copy()
    for par in paragraphs:
        color = get_random_color()
        for line in par.items:
            DrawMore.draw_contours(paragraph_result_image, [line.bbox],
                                   color=color, thickness=2, inplace=True)
    plt.figure(figsize=(10, 12))
    plt.imshow(paragraph_result_image)
