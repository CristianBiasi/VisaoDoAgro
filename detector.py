import time
from pathlib import Path
import cv2
from config import settings
from models import Detection
from proximity import ProximityEstimator
from risk import CollisionRiskEngine, Risk

class YoloDetector:
    def __init__(self):
        self.model = None
        self.proximity = ProximityEstimator()
        self.risk = CollisionRiskEngine()
        self.last_inference = 0.0
        self.last_detections: list[Detection] = []
        self.device = "CPU"

    def _load(self):
        if self.model is not None:
            return
        from ultralytics import YOLO
        self.model = YOLO(settings.model_path)
        self.device = "GPU" if getattr(self.model, "device", None) and str(self.model.device) != "cpu" else "CPU"

    def process(self, frame):
        self._load()
        started = time.perf_counter()
        names = self.model.names
        classes = [i for i, name in names.items() if name in settings.monitored_classes]
        kwargs = dict(source=frame, conf=settings.confidence, iou=settings.iou, imgsz=settings.image_size, classes=classes, verbose=False)
        result = self.model.track(persist=True, **kwargs) if settings.tracking_enabled else self.model.predict(**kwargs)
        result = result[0]
        height, width = frame.shape[:2]
        detections = []
        boxes = result.boxes
        for index in range(len(boxes)):
            xyxy = boxes.xyxy[index].cpu().tolist()
            confidence = float(boxes.conf[index].cpu())
            class_id = int(boxes.cls[index].cpu())
            track_id = int(boxes.id[index].cpu()) if boxes.id is not None else index + 1
            proximity = self.proximity.update(track_id, xyxy, width, height, time.time(), settings.proximity_scale, settings.proximity_smoothing)
            overlap = self.risk.corridor_overlap(xyxy, width, height, settings.corridor_width, settings.corridor_height)
            candidate = self.risk.candidate(proximity.score, proximity.approach_rate, overlap, confidence, settings.medium_threshold, settings.high_threshold, settings.critical_threshold, settings.approach_threshold)
            level = self.risk.update(track_id, candidate, settings.confirmation_frames)
            center_x = (xyxy[0] + xyxy[2]) / 2 / width
            position = "LEFT" if center_x < 0.38 else "RIGHT" if center_x > 0.62 else "CENTER"
            detections.append(Detection(track_id=track_id, object_class=str(names[class_id]), confidence=round(confidence, 3), bbox=[round(v, 1) for v in xyxy], proximity=proximity.score, approach_rate=proximity.approach_rate, position=position, risk=Risk(level).name, corridor_intersection=round(overlap, 2)))
        self.proximity.forget_missing({d.track_id for d in detections})
        self.last_detections = detections
        self.last_inference = (time.perf_counter() - started) * 1000
        if settings.overlay_enabled:
            self._draw(frame, detections)
        return frame

    def _draw(self, frame, detections):
        height, width = frame.shape[:2]
        x1 = int(width * (0.5 - settings.corridor_width / 2)); x2 = int(width * (0.5 + settings.corridor_width / 2))
        y1 = int(height * (0.5 - settings.corridor_height / 2)); y2 = int(height * (0.5 + settings.corridor_height / 2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 190, 220), 2)
        colors = {"SAFE": (100, 190, 120), "LOW": (130, 200, 230), "MEDIUM": (40, 190, 240), "HIGH": (35, 100, 245), "CRITICAL": (40, 40, 255)}
        for detection in detections:
            box = tuple(int(v) for v in detection.bbox)
            color = colors[detection.risk]
            cv2.rectangle(frame, box[:2], box[2:], color, 2)
            label = f"{detection.object_class.upper()} #{detection.track_id} {detection.confidence:.0%} | {detection.risk} {detection.proximity:.0f}%"
            cv2.rectangle(frame, (box[0], max(0, box[1] - 26)), (min(width, box[0] + len(label) * 10), box[1]), color, -1)
            cv2.putText(frame, label, (box[0] + 5, box[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (12, 20, 25), 1, cv2.LINE_AA)
