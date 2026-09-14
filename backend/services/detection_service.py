from typing import Any

import numpy as np

from backend.domain.entities import Detection
from backend.infrastructure.yolo_model import YoloModel


class DetectionService:
    def __init__(self, model: YoloModel) -> None:
        self.model = model

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return self.model.detect(frame)
