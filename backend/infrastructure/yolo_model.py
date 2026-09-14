import logging
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from backend.core.config import settings
from backend.domain.entities import BoundingBox, Detection, ObstacleClass


logger = logging.getLogger(__name__)


class YoloModelError(RuntimeError):
    """Raised when the YOLO model cannot be loaded or used."""


class YoloModel:
    """Adapter that translates ultralytics predictions into domain detections."""

    DEFAULT_CLASS_MAPPING: dict[int, ObstacleClass] = {
        14: ObstacleClass.AVE,
    }

    def __init__(
        self,
        model_path: str | Path | None = None,
        class_mapping: dict[int, ObstacleClass] | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path or settings.yolo_model_path)
        self.class_mapping = (
            self.DEFAULT_CLASS_MAPPING.copy()
            if class_mapping is None
            else class_mapping
        )
        self._frame_counter = 0

        if model is not None:
            self.model = model
            return

        try:
            self.model = YOLO(str(self.model_path))
        except Exception as exc:
            logger.exception("Could not load YOLO model from %s", self.model_path)
            raise YoloModelError(
                f"Could not load YOLO model from '{self.model_path}'"
            ) from exc

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a frame and return mapped domain detections."""
        self._validate_frame(frame)
        frame_id = self._next_frame_id()

        try:
            results = self.model(frame, verbose=False)
            return self._to_detections(results, frame_id)
        except YoloModelError:
            raise
        except Exception as exc:
            logger.exception("Could not run YOLO inference for %s", frame_id)
            raise YoloModelError("Could not run YOLO inference") from exc

    def _validate_frame(self, frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.error("Invalid frame: expected a non-empty numpy array")
            raise YoloModelError("Frame must be a non-empty numpy array")
        if frame.ndim not in (2, 3):
            logger.error("Invalid frame shape: %s", frame.shape)
            raise YoloModelError("Frame must have 2 or 3 dimensions")

    def _next_frame_id(self) -> str:
        self._frame_counter += 1
        return f"frame-{self._frame_counter}"

    def _to_detections(self, results: Any, frame_id: str) -> list[Detection]:
        detections: list[Detection] = []

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            coordinates = self._to_list(boxes.xyxy)
            confidences = self._to_list(boxes.conf)
            class_ids = self._to_list(boxes.cls)

            for box, confidence, class_id in zip(
                coordinates, confidences, class_ids, strict=False
            ):
                obstacle_class = self.class_mapping.get(int(class_id))
                if obstacle_class is None:
                    continue

                detections.append(
                    Detection(
                        obstacle_class=obstacle_class,
                        confidence=float(confidence),
                        bounding_box=BoundingBox(
                            x_min=float(box[0]),
                            y_min=float(box[1]),
                            x_max=float(box[2]),
                            y_max=float(box[3]),
                        ),
                        frame_id=frame_id,
                    )
                )

        return detections

    @staticmethod
    def _to_list(values: Any) -> list[Any]:
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "tolist"):
            values = values.tolist()
        return values
