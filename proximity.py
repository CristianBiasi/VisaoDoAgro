from collections import defaultdict, deque
from dataclasses import dataclass
import math

@dataclass
class Proximity:
    score: float
    approach_rate: float

class ProximityEstimator:
    """Visual score provider. Scores are relative, never physical distances."""

    def __init__(self, history_size: int = 12, smoothing: float = 0.35):
        self.history: dict[int, deque[tuple[float, float]]] = defaultdict(lambda: deque(maxlen=history_size))
        self.smoothing = smoothing
        self.smoothed: dict[int, float] = {}

    def update(self, track_id: int, bbox: list[float], frame_width: int, frame_height: int, timestamp: float, score_scale: float = 150.0, smoothing: float | None = None) -> Proximity:
        x1, y1, x2, y2 = bbox
        ratio = max(0.0, min(1.0, (max(0.0, x2 - x1) * max(0.0, y2 - y1)) / max(1, frame_width * frame_height)))
        raw_score = min(100.0, math.sqrt(ratio) * score_scale)
        previous = self.smoothed.get(track_id, raw_score)
        active_smoothing = self.smoothing if smoothing is None else smoothing
        score = previous + active_smoothing * (raw_score - previous)
        self.smoothed[track_id] = score
        history = self.history[track_id]
        history.append((timestamp, score))
        rate = 0.0
        if len(history) >= 2:
            old_time, old_score = history[0]
            elapsed = max(0.001, timestamp - old_time)
            rate = (score - old_score) / elapsed
        return Proximity(round(score, 1), round(rate, 1))

    def forget_missing(self, visible_ids: set[int]) -> None:
        for track_id in list(self.history):
            if track_id not in visible_ids:
                del self.history[track_id]
                self.smoothed.pop(track_id, None)
