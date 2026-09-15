"""Compose detection, visual proximity and risk without drawing or network I/O."""
import time
from dataclasses import dataclass

from models import Detection
from proximity import ProximityEstimator
from risk import CollisionRiskEngine


@dataclass
class FrameResult:
    detections: list[Detection]
    width: int
    height: int
    inference_ms: float
    processing_ms: float
    received_at: float


class DetectionProcessor:
    def __init__(self, detector, proximity=None):
        self.detector = detector
        # A future depth implementation can be injected here; no fake meters.
        self.proximity = proximity if proximity is not None else ProximityEstimator()
        self.risk = CollisionRiskEngine()
        self._settings_key = None
        self._model_generation = None

    def process(self, frame, settings, received_at):
        started = time.perf_counter()
        key = (settings.model_path, settings.device, settings.half_precision,
               settings.tracking_enabled, tuple(settings.monitored_classes),
               settings.proximity_scale, settings.proximity_smoothing,
               settings.medium_threshold, settings.high_threshold,
               settings.critical_threshold, settings.approach_threshold,
               settings.corridor_width, settings.corridor_height, settings.confirmation_frames)
        if key != self._settings_key:
            self.proximity.forget_missing(set())
            self.risk.forget_missing(set())
            self._settings_key = key
        image = frame.to_ndarray(format="bgr24")
        height, width = image.shape[:2]
        objects = self.detector.detect(image, settings)
        generation = getattr(self.detector, "generation", None)
        if generation != self._model_generation:
            self.proximity.forget_missing(set())
            self.risk.forget_missing(set())
            self._model_generation = generation
        tracked_ids = {item.track_id for item in objects if item.tracked}
        self.proximity.forget_missing(tracked_ids)
        self.risk.forget_missing(tracked_ids)
        detections = []
        # Predict-only IDs are frame-local: do not calculate approach across unrelated objects.
        if not settings.tracking_enabled:
            self.proximity.forget_missing(set())
            self.risk.forget_missing(set())
        for item in objects:
            proximity = self.proximity.update(
                item.track_id, item.bbox, width, height, received_at,
                settings.proximity_scale, settings.proximity_smoothing,
            )
            overlap = self.risk.corridor_overlap(
                item.bbox, width, height, settings.corridor_width, settings.corridor_height,
            )
            candidate = self.risk.candidate(
                proximity.score, proximity.approach_rate, overlap, item.confidence,
                settings.medium_threshold, settings.high_threshold,
                settings.critical_threshold, settings.approach_threshold,
            )
            level = self.risk.update(item.track_id, candidate, settings.confirmation_frames) if item.tracked else candidate
            center = (item.bbox[0] + item.bbox[2]) / 2 / width
            position = "LEFT" if center < 0.38 else "RIGHT" if center > 0.62 else "CENTER"
            detections.append(Detection(
                track_id=item.track_id, object_class=item.object_class,
                confidence=round(item.confidence, 3), bbox=[round(value, 1) for value in item.bbox],
                proximity=proximity.score, approach_rate=proximity.approach_rate,
                position=position, risk=level.name, corridor_intersection=round(overlap, 2),
            ))
        visible = tracked_ids
        self.proximity.forget_missing(visible)
        self.risk.forget_missing(visible)
        return FrameResult(detections, width, height, self.detector.last_inference,
                           (time.perf_counter() - started) * 1000, received_at)
