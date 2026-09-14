import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backend.domain.entities import BoundingBox, Detection, ObstacleClass


@dataclass(frozen=True)
class MetricsReport:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    map_simplified: float
    false_positive_rate: float
    false_negative_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class MetricsService:
    def __init__(self, iou_threshold: float = 0.5) -> None:
        if not 0 <= iou_threshold <= 1:
            raise ValueError("IoU threshold must be between 0 and 1")
        self.iou_threshold = iou_threshold

    def calculate(
        self,
        predictions: list[Detection],
        annotation_path: str | Path,
    ) -> MetricsReport:
        ground_truth = self._load_annotations(annotation_path)
        matched_ground_truth: set[int] = set()
        true_positives = 0

        for prediction in predictions:
            best_match: tuple[float, int] | None = None
            for index, expected in enumerate(ground_truth):
                if index in matched_ground_truth:
                    continue
                if prediction.frame_id != expected.frame_id:
                    continue
                if prediction.obstacle_class is not expected.obstacle_class:
                    continue

                overlap = self._intersection_over_union(
                    prediction.bounding_box,
                    expected.bounding_box,
                )
                if overlap >= self.iou_threshold and (
                    best_match is None or overlap > best_match[0]
                ):
                    best_match = (overlap, index)

            if best_match is None:
                continue
            true_positives += 1
            matched_ground_truth.add(best_match[1])

        false_positives = len(predictions) - true_positives
        false_negatives = len(ground_truth) - true_positives
        precision = self._ratio(true_positives, true_positives + false_positives)
        recall = self._ratio(true_positives, true_positives + false_negatives)

        return MetricsReport(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            map_simplified=precision,
            false_positive_rate=self._ratio(
                false_positives,
                true_positives + false_positives,
            ),
            false_negative_rate=self._ratio(
                false_negatives,
                true_positives + false_negatives,
            ),
        )

    @staticmethod
    def _load_annotations(path: str | Path) -> list[Detection]:
        annotation_path = Path(path)
        if annotation_path.suffix.lower() == ".csv":
            with annotation_path.open(newline="", encoding="utf-8") as file:
                return [
                    MetricsService._annotation_to_detection(row)
                    for row in csv.DictReader(file)
                ]

        with annotation_path.open(encoding="utf-8") as file:
            content: Any = json.load(file)
        if isinstance(content, dict):
            content = content.get("annotations", [])
        if not isinstance(content, list):
            raise ValueError("Annotation file must contain a list of annotations")
        return [MetricsService._annotation_to_detection(item) for item in content]

    @staticmethod
    def _annotation_to_detection(annotation: dict[str, Any]) -> Detection:
        box = annotation.get("bounding_box") or annotation
        return Detection(
            obstacle_class=ObstacleClass(annotation["obstacle_class"]),
            confidence=1.0,
            bounding_box=BoundingBox(
                x_min=float(box["x_min"]),
                y_min=float(box["y_min"]),
                x_max=float(box["x_max"]),
                y_max=float(box["y_max"]),
            ),
            frame_id=str(annotation["frame_id"]),
        )

    @staticmethod
    def _intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
        intersection_x_min = max(first.x_min, second.x_min)
        intersection_y_min = max(first.y_min, second.y_min)
        intersection_x_max = min(first.x_max, second.x_max)
        intersection_y_max = min(first.y_max, second.y_max)
        intersection_width = max(0.0, intersection_x_max - intersection_x_min)
        intersection_height = max(0.0, intersection_y_max - intersection_y_min)
        intersection = intersection_width * intersection_height

        first_area = (first.x_max - first.x_min) * (first.y_max - first.y_min)
        second_area = (second.x_max - second.x_min) * (second.y_max - second.y_min)
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0
