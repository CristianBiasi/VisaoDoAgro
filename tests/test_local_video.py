"""Automated tests for LocalVideoProcessor: decoupled playback, seek, loop, speed, metrics."""
import asyncio
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from config import Settings
from detector import ObjectDetection
from local_video import LocalVideoProcessor


def create_synthetic_video(path: Path, frame_count: int = 30, fps: float = 10.0, width: int = 64, height: int = 48) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(frame_count):
        frame = np.full((height, width, 3), (i * 8) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


class FakeDetector:
    def __init__(self, detections=None, delay=0.0):
        self.device = "cpu"
        self.half = False
        self.last_inference = 5.0
        self.loaded_model = "test.pt"
        self._detections = detections or []
        self.delay = delay
        self.reset_tracking = Mock()

    def detect(self, image, settings):
        if self.delay > 0:
            time.sleep(self.delay)
        return self._detections


class LocalVideoProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.video_path = Path(self.temp_dir.name) / "test_synth.mp4"
        create_synthetic_video(self.video_path, frame_count=40, fps=15.0)
        self.settings = Settings(inference_fps=30)
        self.video_clients = set()
        self.results = []
        self.errors = []

    async def asyncTearDown(self):
        self.temp_dir.cleanup()

    def make_processor(self, detector=None, speed=1.0, loop=True):
        det = detector or FakeDetector()
        proc = LocalVideoProcessor(
            detector=det,
            get_settings=lambda: self.settings,
            on_result=lambda r, m: self.results.append((r, m)),
            on_error=lambda e: self.errors.append(e),
            video_clients=self.video_clients,
            get_view_mode=lambda: "AI",
        )
        proc.speed = speed
        proc.loop = loop
        return proc

    async def test_pause_and_resume(self):
        proc = self.make_processor()
        proc.start(str(self.video_path))
        await asyncio.sleep(0.3)
        self.assertTrue(proc.running)

        proc.pause()
        self.assertTrue(proc.paused)
        f1 = proc.frame_index
        await asyncio.sleep(0.3)
        f2 = proc.frame_index
        self.assertLessEqual(abs(f2 - f1), 1)

        proc.play()
        self.assertFalse(proc.paused)
        await asyncio.sleep(0.3)
        f3 = proc.frame_index
        self.assertGreater(f3, f2)

        await proc.stop()
        self.assertFalse(proc.running)

    async def test_seek_and_resets_tracking(self):
        det = FakeDetector()
        proc = self.make_processor(detector=det)
        proc.start(str(self.video_path))
        await asyncio.sleep(0.2)

        det.reset_tracking.reset_mock()
        # Seek to 1.0 second (at 15 fps = frame 15)
        proc.seek(1.0)
        await asyncio.sleep(0.2)

        self.assertGreaterEqual(proc.frame_index, 14)
        det.reset_tracking.assert_called()

        await proc.stop()

    async def test_loop_true_and_false(self):
        # loop = False terminates when video finishes
        short_video = Path(self.temp_dir.name) / "short.mp4"
        create_synthetic_video(short_video, frame_count=6, fps=20.0)

        proc_no_loop = self.make_processor(loop=False)
        proc_no_loop.start(str(short_video))
        await asyncio.sleep(0.6)
        self.assertTrue(proc_no_loop.finished)
        await proc_no_loop.stop()

        # loop = True resets to frame 0 and continues
        proc_loop = self.make_processor(loop=True)
        proc_loop.start(str(short_video))
        await asyncio.sleep(0.6)
        self.assertTrue(proc_loop.running)
        self.assertFalse(proc_loop.finished)
        await proc_loop.stop()

    async def test_speed_advances_more_frames(self):
        proc_normal = self.make_processor(speed=1.0)
        proc_normal.start(str(self.video_path))
        await asyncio.sleep(0.4)
        normal_frames = proc_normal.frame_index
        await proc_normal.stop()

        proc_fast = self.make_processor(speed=2.0)
        proc_fast.start(str(self.video_path))
        await asyncio.sleep(0.4)
        fast_frames = proc_fast.frame_index
        await proc_fast.stop()

        self.assertGreater(fast_frames, normal_frames)

    async def test_metrics_video_fps_differs_from_ai_fps(self):
        # Detector delay of 80ms limits AI to <= 12 FPS, while video runs at 25 FPS
        video_25 = Path(self.temp_dir.name) / "fps25.mp4"
        create_synthetic_video(video_25, frame_count=50, fps=25.0)

        slow_detector = FakeDetector(delay=0.08)
        proc = self.make_processor(detector=slow_detector)
        proc.start(str(video_25))
        await asyncio.sleep(0.7)

        metrics = proc.metrics()
        self.assertIn("video_fps", metrics)
        self.assertIn("ai_fps", metrics)
        self.assertGreater(metrics["video_fps"], metrics["ai_fps"])

        await proc.stop()

    async def test_high_detection_creates_alert_marks_without_duplicates(self):
        high_det = ObjectDetection(
            track_id=42, object_class="person", confidence=0.95,
            bbox=[10, 10, 50, 50], tracked=True
        )
        detector = FakeDetector(detections=[high_det])
        proc = self.make_processor(detector=detector)
        proc.start(str(self.video_path))

        await asyncio.sleep(0.5)
        await proc.stop()

        marks = [m for m in proc.alert_marks if m.get("track_id") == 42]
        self.assertGreaterEqual(len(marks), 1)
        # Verify no two marks for the same track_id within 1 second
        for i in range(len(marks) - 1):
            diff = abs(marks[i + 1]["t"] - marks[i]["t"])
            self.assertGreaterEqual(diff, 1.0)

        state = proc.playback_state()
        self.assertIn("alert_marks", state)
        self.assertIn("position_s", state)
        self.assertIn("duration_s", state)
