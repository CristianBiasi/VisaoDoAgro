from collections import defaultdict
from enum import IntEnum

class Risk(IntEnum):
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class CollisionRiskEngine:
    def __init__(self):
        self.pending: dict[int, tuple[Risk, int]] = defaultdict(lambda: (Risk.SAFE, 0))

    @staticmethod
    def corridor_overlap(bbox: list[float], width: int, height: int, corridor_width: float, corridor_height: float) -> float:
        cx, cy = width / 2, height / 2
        cw, ch = width * corridor_width / 2, height * corridor_height / 2
        x1, y1, x2, y2 = bbox
        ix = max(0.0, min(x2, cx + cw) - max(x1, cx - cw))
        iy = max(0.0, min(y2, cy + ch) - max(y1, cy - ch))
        intersection = ix * iy
        return min(1.0, intersection / max(1.0, (x2 - x1) * (y2 - y1)))

    def candidate(self, proximity: float, approach: float, overlap: float, confidence: float, medium: float, high: float, critical: float, approach_threshold: float) -> Risk:
        danger = proximity + overlap * 24 + max(0.0, approach) * 1.4 + max(0.0, confidence - 0.5) * 8
        if danger >= critical + 24 or (proximity >= critical and overlap >= 0.35 and approach >= approach_threshold):
            return Risk.CRITICAL
        if danger >= high or (proximity >= high and overlap >= 0.2):
            return Risk.HIGH
        if danger >= medium or (proximity >= medium and overlap >= 0.1):
            return Risk.MEDIUM
        return Risk.LOW if proximity >= 12 else Risk.SAFE

    def update(self, track_id: int, candidate: Risk, confirmation_frames: int) -> Risk:
        current, count = self.pending[track_id]
        if candidate > current:
            count += 1
            if count >= confirmation_frames:
                current, count = candidate, 0
        elif candidate < current:
            count += 1
            if count >= confirmation_frames + 1:
                current, count = candidate, 0
        else:
            count = 0
        self.pending[track_id] = (current, count)
        return current
