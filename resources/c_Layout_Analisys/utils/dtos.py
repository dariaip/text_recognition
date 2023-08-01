from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Word:
    bbox: np.ndarray
    label: str


@dataclass
class Line:
    bbox: Optional[np.ndarray] = None
    normalized_bbox: Optional[np.ndarray] = None
    items: List[Word] = field(default_factory=list)
    label: Optional[str] = None


@dataclass
class Paragraph:
    bbox: Optional[np.ndarray] = None
    items: List[Line] = field(default_factory=list)
    label: Optional[str] = None
