"""Ultralytics adapter: loading, device selection and structured detections."""
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from config import ROOT, Settings

logger = logging.getLogger(__name__)


@dataclass
class ObjectDetection:
    track_id: int
    object_class: str
    confidence: float
    bbox: list[float]
    tracked: bool


def select_device(requested: str, torch_module=None) -> str:
    """Keep backend selection here so the rest of the app is device agnostic."""
    if torch_module is None:
        import torch as torch_module
    if requested == "cpu":
        return "cpu"
    if torch_module.cuda.is_available():
        index = int(requested.split(":")[1]) if ":" in requested else 0
        if index < torch_module.cuda.device_count():
            return f"cuda:{index}"
    if requested != "auto":
        logger.warning("Requested device %s unavailable; using CPU", requested)
    return "cpu"


class YoloDetector:
    """Used by one serial inference worker; never reload a model mid-inference."""

    def __init__(self, model_factory=None, device_selector=select_device):
        self.model = None
        self.generation = 0
        self._factory = model_factory
        self._select_device = device_selector
        self._load_key = None
        self._class_key = None
        self._class_ids = None
        self.available_classes = {}
        self.missing_classes = []
        self.device = "UNAVAILABLE"
        self.half = False
        self.loaded_model = None
        self.last_inference = 0.0
        self.last_detections = []

    def reset_tracking(self):
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []):
            tracker.reset()

    @staticmethod
    def configuration_key(settings: Settings):
        # A fresh model also removes tracking callbacks when switching to predict.
        return (settings.model_path, settings.device, settings.half_precision, settings.tracking_enabled)

    def needs_reload(self, settings: Settings) -> bool:
        return self._load_key != self.configuration_key(settings)

    def _load(self, settings: Settings):
        key = self.configuration_key(settings)
        if self.model is not None and key == self._load_key:
            return
        path = Path(settings.model_path)
        resolved = path if path.is_absolute() else ROOT / path
        # Only the bundled generic model may be downloaded automatically.
        if not resolved.is_file() and settings.model_path != "yolo11n.pt":
            raise FileNotFoundError(f"Model not found: {resolved}")
        factory = self._factory
        if factory is None:
            from ultralytics import YOLO
            factory = YOLO
        model = factory(str(resolved))
        names = model.names
        names = dict(enumerate(names)) if isinstance(names, (list, tuple)) else dict(names)
        device = self._select_device(settings.device)
        # Commit only after loading succeeds; a failed switch is retried, not silently ignored.
        self.model = model
        self.generation += 1
        self.available_classes = names
        self.device = device
        self.half = settings.half_precision and device.startswith("cuda")
        self.loaded_model = settings.model_path
        self._load_key = key
        self._class_key = None
        logger.info("Loaded %s on %s (FP16=%s)", settings.model_path, device, self.half)

    def _classes(self, settings: Settings):
        key = (tuple(settings.monitored_classes), settings.tracking_enabled)
        if key != self._class_key:
            self.reset_tracking()
            wanted = set(settings.monitored_classes)
            self.missing_classes = sorted(wanted - set(self.available_classes.values()))
            if self.missing_classes:
                logger.warning("Classes absent from %s: %s", self.loaded_model, self.missing_classes)
            self._class_ids = [index for index, name in self.available_classes.items() if name in wanted]
            self._class_key = key
        # An empty user list explicitly selects all model classes.
        return self._class_ids if settings.monitored_classes else None

    def detect(self, frame, settings: Settings) -> list[ObjectDetection]:
        self._load(settings)
        classes = self._classes(settings)
        if classes == []:
            self.last_inference = 0.0
            self.last_detections = []
            return []
        kwargs = dict(source=frame, conf=settings.confidence, iou=settings.iou,
                      imgsz=settings.image_size, classes=classes, verbose=False,
                      device=self.device, half=self.half)
        started = time.perf_counter()
        try:
            results = self._infer(settings, kwargs)
        except RuntimeError:
            if not self.device.startswith("cuda"):
                raise
            logger.exception("CUDA inference failed; retrying on CPU without FP16")
            self.model = None
            self._load(settings.model_copy(update={"device": "cpu", "half_precision": False}))
            self._classes(settings)
            # Keep the fallback until a model/device/precision setting changes.
            self._load_key = self.configuration_key(settings)
            kwargs.update(device="cpu", half=False)
            results = self._infer(settings, kwargs)
        boxes = results[0].boxes
        detections = []
        if boxes is not None:
            # One transfer/synchronization for every box and all its fields.
            rows = boxes.data.cpu().tolist()
            for index, row in enumerate(rows):
                tracked = len(row) == 7
                track_id = int(row[4]) if tracked else -(index + 1)
                confidence, class_id = row[-2:]
                detections.append(ObjectDetection(
                    track_id, str(self.available_classes[int(class_id)]),
                    float(confidence), list(row[:4]), tracked,
                ))
        self.last_inference = (time.perf_counter() - started) * 1000
        self.last_detections = detections
        return detections

    def _infer(self, settings, kwargs):
        if settings.tracking_enabled:
            return self.model.track(persist=True, **kwargs)
        return self.model.predict(**kwargs)
