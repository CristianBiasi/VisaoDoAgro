from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


class VideoSourceError(RuntimeError):
    """Raised when a video source cannot be opened or read."""


class VideoSource:
    """OpenCV-backed frame source for local video files."""

    def __init__(self, source: str | Path) -> None:
        self.source = str(source)

    def frames(self) -> Iterator[np.ndarray]:
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(f"Could not open video source '{self.source}'")

        try:
            while True:
                has_frame, frame = capture.read()
                if not has_frame:
                    break
                yield frame
        except Exception as exc:
            raise VideoSourceError(
                f"Could not read video source '{self.source}'"
            ) from exc
        finally:
            capture.release()
