"""Draw an overlay on the current video frame, independently of inference."""
import cv2

COLORS = {"SAFE": (100, 190, 120), "LOW": (130, 200, 230),
          "MEDIUM": (40, 190, 240), "HIGH": (35, 100, 245), "CRITICAL": (40, 40, 255)}


def draw_overlay(frame, detections, settings):
    height, width = frame.shape[:2]
    left = int(width * (0.5 - settings.corridor_width / 2))
    right = int(width * (0.5 + settings.corridor_width / 2))
    top = int(height * (0.5 - settings.corridor_height / 2))
    bottom = int(height * (0.5 + settings.corridor_height / 2))
    cv2.rectangle(frame, (left, top), (right, bottom), (40, 190, 220), 2)
    for detection in detections:
        box = tuple(int(value) for value in detection.bbox)
        color = COLORS[detection.risk]
        cv2.rectangle(frame, box[:2], box[2:], color, 2)
        label = (f"{detection.object_class.upper()} #{detection.track_id} "
                 f"{detection.confidence:.0%} | {detection.risk} {detection.proximity:.0f}%")
        label_top = max(18, box[1])
        cv2.rectangle(frame, (box[0], max(0, label_top - 26)),
                      (min(width, box[0] + len(label) * 10), label_top), color, -1)
        cv2.putText(frame, label, (box[0] + 5, label_top - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (12, 20, 25), 1, cv2.LINE_AA)
    return frame
