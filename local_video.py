"""Process a locally uploaded video file frame-by-frame using OpenCV + YOLO.

Bypasses WebRTC entirely — the browser uploads the file via HTTP, and
the server reads it with cv2.VideoCapture, running the same detection
pipeline used for the live phone camera.
"""
import asyncio
import logging
import time

import cv2
from av import VideoFrame

from overlay import draw_overlay
from processing import DetectionProcessor, FrameResult
from video_pipeline import LatestFrame, RateMeter

logger = logging.getLogger(__name__)


class LocalVideoProcessor:
    """Manages background processing of a single uploaded video file."""

    def __init__(self, detector, get_settings, on_result, on_error, video_clients):
        self.detector = detector
        self.processor = DetectionProcessor(detector)
        self.get_settings = get_settings
        self.on_result = on_result
        self.on_error = on_error
        self.video_clients = video_clients
        self.ai_rate = RateMeter()
        self._task: asyncio.Task | None = None
        self._stopped = False
        self.dropped_frames = 0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, video_path: str):
        """Start processing the video in a background task."""
        self._stopped = False
        self.dropped_frames = 0
        self.ai_rate = RateMeter()
        self.detector.reset_tracking()
        self._task = asyncio.create_task(self._run(video_path))

    async def stop(self):
        """Stop current processing."""
        self._stopped = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None

    def metrics(self):
        now = time.perf_counter()
        return {
            "ai_fps": round(self.ai_rate.value(now), 1),
            "video_fps": round(self.ai_rate.value(now), 1),
            "inference_ms": round(self.detector.last_inference, 1),
            "dropped_frames": self.dropped_frames,
            "device": self.detector.device,
            "backend": "Ultralytics / PyTorch (Local Video)",
            "half_precision": self.detector.half,
        }

    async def _run(self, video_path: str):
        """Main loop: read video → detect → overlay → JPEG → push to clients."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error("Could not open video: %s", video_path)
                return

            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_delay = 1.0 / video_fps
            logger.info(
                "Local video started: %s (%.1f FPS, %d frames)",
                video_path, video_fps, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            )

            while not self._stopped:
                frame_start = time.perf_counter()

                # Read frame in a thread to avoid blocking the event loop
                success, bgr_frame = await asyncio.to_thread(cap.read)
                if not success or self._stopped:
                    break

                settings = self.get_settings()
                received_at = time.perf_counter()

                # Convert to av.VideoFrame so the existing processor works unchanged
                av_frame = VideoFrame.from_ndarray(bgr_frame, format="bgr24")

                try:
                    result: FrameResult = await asyncio.to_thread(
                        self.processor.process, av_frame, settings, received_at,
                    )
                except Exception as exc:
                    logger.exception("Local video inference failed")
                    self.on_error(exc)
                    await asyncio.sleep(1)
                    continue

                if self._stopped:
                    break

                now = time.perf_counter()
                self.ai_rate.tick(now)
                self.on_result(result, self.metrics())

                # Render overlay and send JPEG to dashboard video clients
                image = bgr_frame.copy()
                if settings.overlay_enabled and result.detections:
                    draw_overlay(image, result.detections, settings)

                success_enc, encoded = cv2.imencode(
                    ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82]
                )
                if success_enc:
                    jpeg_bytes = encoded.tobytes()
                    for slot in tuple(self.video_clients):
                        slot.put(jpeg_bytes)

                # Pace playback to match original video FPS
                elapsed = time.perf_counter() - frame_start
                sleep_time = frame_delay - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    self.dropped_frames += 1

        except asyncio.CancelledError:
            logger.info("Local video processing cancelled")
        except Exception:
            logger.exception("Local video processing error")
        finally:
            cap.release()
            logger.info("Local video processing finished")
