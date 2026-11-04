"""Process a locally uploaded video file frame-by-frame using OpenCV + YOLO.

Decouples video playback from inference so local video playback runs smoothly
with full controls (pause, seek, speed, loop) and real-time metrics.
"""
import asyncio
from dataclasses import dataclass
import logging
import time

import cv2
import numpy as np
from av import VideoFrame

from overlay import draw_overlay
from processing import DetectionProcessor, FrameResult
from video_pipeline import LatestFrame, MediaStreamError, RateMeter, MAX_OVERLAY_AGE

logger = logging.getLogger(__name__)

MAX_JPEG_WIDTH = 1280
JPEG_QUALITY = 75
ALLOWED_SPEEDS = (0.25, 0.5, 1.0, 1.5, 2.0)
PAUSE_RERENDER_INTERVAL = 0.2  # ~5 FPS re-rendering when paused


@dataclass
class VideoInferItem:
    frame: VideoFrame
    frame_index: int
    received_at: float


class LocalVideoProcessor:
    """Manages playback and decoupled inference of a local or demo video file."""

    def __init__(self, detector, get_settings, on_result, on_error, video_clients, get_view_mode=lambda: "AI"):
        self.detector = detector
        self.processor = DetectionProcessor(detector)
        self.get_settings = get_settings
        self.get_view_mode = get_view_mode
        self.on_result = on_result
        self.on_error = on_error
        self.video_clients = video_clients

        self.paused: bool = False
        self.speed: float = 1.0
        self.loop: bool = True
        self.frame_index: int = 0
        self.frame_count: int = 0
        self.video_fps: float = 30.0
        self.source_name: str = "video"
        self._seek_to: int | None = None
        self._last_raw: np.ndarray | None = None
        self.alert_marks: list[dict] = []
        self.dropped_frames: int = 0
        self.finished: bool = False

        self.latest_result: FrameResult | None = None
        self.latest_result_frame_index: int | None = None
        self.result_settings = None

        self.ai_rate = RateMeter()
        self.video_rate = RateMeter()
        self.inference_slot = LatestFrame()
        self.overlay_ms: float = 0.0
        self.jpeg_ms: float = 0.0

        self._playback_task: asyncio.Task | None = None
        self._infer_task: asyncio.Task | None = None
        self._stopped: bool = False

    @property
    def running(self) -> bool:
        return self._playback_task is not None and not self._playback_task.done()

    def start(self, video_path: str):
        """Start processing the video in background playback and inference tasks."""
        self._stopped = False
        self.finished = False
        self.paused = False
        self.frame_index = 0
        self.dropped_frames = 0
        self.ai_rate = RateMeter()
        self.video_rate = RateMeter()
        self.inference_slot = LatestFrame()
        self.detector.reset_tracking()
        self.processor.reset()
        self._playback_task = asyncio.create_task(self._playback(video_path))
        self._infer_task = asyncio.create_task(self._infer())

    async def stop(self):
        """Stop playback and inference tasks."""
        self._stopped = True
        self.inference_slot.close()
        for task in (self._playback_task, self._infer_task):
            if task and not task.done():
                task.cancel()
        tasks = [t for t in (self._playback_task, self._infer_task) if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._playback_task = None
        self._infer_task = None

    def seek(self, seconds: float):
        fps = max(1.0, self.video_fps)
        self._seek_to = max(0, min(max(0, self.frame_count - 1), int(seconds * fps)))

    def seek_relative(self, delta_seconds: float):
        fps = max(1.0, self.video_fps)
        current_s = self.frame_index / fps
        self.seek(current_s + delta_seconds)

    def set_speed(self, speed: float):
        if speed in ALLOWED_SPEEDS:
            self.speed = float(speed)

    def set_loop(self, loop: bool):
        self.loop = bool(loop)

    def play(self):
        self.paused = False

    def pause(self):
        self.paused = True

    def toggle(self):
        self.paused = not self.paused

    def playback_state(self) -> dict:
        fps = max(1.0, self.video_fps)
        return {
            "active": self.running,
            "paused": self.paused,
            "speed": self.speed,
            "loop": self.loop,
            "position_s": round(self.frame_index / fps, 2),
            "duration_s": round(self.frame_count / fps, 2),
            "source_name": self.source_name,
            "alert_marks": list(self.alert_marks[-200:]),
        }

    def metrics(self, now=None):
        now = time.perf_counter() if now is None else now
        result = self.latest_result
        return {
            "ai_fps": round(self.ai_rate.value(now), 1),
            "video_fps": round(self.video_rate.value(now), 1),
            "server_detection_ms": round((now - result.received_at) * 1000, 1) if result else None,
            "result_age_ms": round((now - result.received_at) * 1000, 1) if result else None,
            "inference_ms": round(result.inference_ms, 1) if result else round(self.detector.last_inference, 1),
            "processing_ms": round(result.processing_ms, 1) if result else 0.0,
            "overlay_ms": round(self.overlay_ms, 1),
            "jpeg_ms": round(self.jpeg_ms, 1),
            "dropped_frames": self.dropped_frames,
            "device": self.detector.device,
            "backend": "Ultralytics / PyTorch (Local Video)",
            "half_precision": self.detector.half,
        }

    async def _infer(self):
        """Worker loop: decoupled AI inference paced by settings.inference_fps."""
        next_inference = 0.0
        try:
            while not self._stopped:
                await asyncio.sleep(max(0, next_inference - time.perf_counter()))
                item: VideoInferItem = await self.inference_slot.get()
                if self._stopped:
                    break

                configuration = self.get_settings()
                started = time.perf_counter()
                try:
                    result: FrameResult = await asyncio.to_thread(
                        self.processor.process, item.frame, configuration, item.received_at
                    )
                    if self._stopped:
                        break
                    if configuration is not self.get_settings():
                        continue

                    self.latest_result = result
                    self.latest_result_frame_index = item.frame_index
                    self.result_settings = configuration
                    now = time.perf_counter()
                    self.ai_rate.tick(now)

                    # Timeline alert markers
                    t_sec = round(item.frame_index / max(1.0, self.video_fps), 2)
                    for det in result.detections:
                        if det.risk in ("HIGH", "CRITICAL"):
                            recent = any(
                                m.get("track_id") == det.track_id and abs(m["t"] - t_sec) < 1.0
                                for m in self.alert_marks
                            )
                            if not recent:
                                self.alert_marks.append({
                                    "t": t_sec,
                                    "risk": det.risk,
                                    "track_id": det.track_id,
                                    "class": det.object_class,
                                })
                                if len(self.alert_marks) > 200:
                                    self.alert_marks.pop(0)

                    self.on_result(result, self.metrics(now))
                except Exception as exc:
                    self.latest_result = None
                    self.latest_result_frame_index = None
                    self.result_settings = None
                    logger.exception("Local video inference failed")
                    self.on_error(exc)
                    next_inference = time.perf_counter() + 1.0
                    continue

                next_inference = started + 1.0 / max(1, configuration.inference_fps)
        except (asyncio.CancelledError, MediaStreamError):
            pass
        except Exception:
            logger.exception("Local video inference loop error")

    async def _render_and_publish(self, bgr_frame: np.ndarray):
        """Render overlay (if active and fresh) and publish scaled JPEG to clients."""
        started = time.perf_counter()
        settings = self.get_settings()
        view_mode = self.get_view_mode()
        result = self.latest_result

        # Invalidate old result based on age or frame difference
        if result:
            now = time.perf_counter()
            diff_frames = abs(self.frame_index - (self.latest_result_frame_index or 0))
            if (now - result.received_at > MAX_OVERLAY_AGE) or (diff_frames > self.video_fps * 1.0):
                result = None

        if result and settings.overlay_enabled and view_mode == "AI":
            image = bgr_frame.copy()
            height, width = image.shape[:2]
            detections = result.detections
            if (width, height) != (result.width, result.height):
                detections = [item.model_copy(update={"bbox": [
                    value * (width / result.width if index % 2 == 0 else height / result.height)
                    for index, value in enumerate(item.bbox)
                ]}) for item in detections]
            draw_overlay(image, detections, settings)
        else:
            image = bgr_frame

        self.overlay_ms = (time.perf_counter() - started) * 1000

        # Downscale JPEG if width exceeds MAX_JPEG_WIDTH
        h, w = image.shape[:2]
        if w > MAX_JPEG_WIDTH:
            new_h = int(h * (MAX_JPEG_WIDTH / w))
            image = cv2.resize(image, (MAX_JPEG_WIDTH, new_h), interpolation=cv2.INTER_AREA)

        t_jpeg = time.perf_counter()
        success_enc, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if success_enc:
            jpeg_bytes = encoded.tobytes()
            for slot in tuple(self.video_clients):
                slot.put(jpeg_bytes)
        self.jpeg_ms = (time.perf_counter() - t_jpeg) * 1000
        self.video_rate.tick(time.perf_counter())

    async def _playback(self, video_path: str):
        """Playback loop: reads frames, paces via clock, handles seek/pause/loop."""
        cap = None
        try:
            cap = await asyncio.to_thread(cv2.VideoCapture, video_path)
            if not cap.isOpened():
                logger.error("Could not open video: %s", video_path)
                self.finished = True
                return

            self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            self.frame_index = 0
            logger.info(
                "Local video playback started: %s (%.1f FPS, %d frames)",
                video_path, self.video_fps, self.frame_count,
            )

            clock_time = time.perf_counter()
            clock_frame = 0

            while not self._stopped:
                # 1. Apply pending seek
                if self._seek_to is not None:
                    target = max(0, min(max(0, self.frame_count - 1), self._seek_to))
                    self._seek_to = None
                    await asyncio.to_thread(cap.set, cv2.CAP_PROP_POS_FRAMES, target)
                    self.frame_index = target
                    self.detector.reset_tracking()
                    self.processor.reset()
                    self.latest_result = None
                    self.latest_result_frame_index = None
                    success, bgr_frame = await asyncio.to_thread(cap.read)
                    if success:
                        self._last_raw = bgr_frame
                        self.frame_index += 1
                        # Push seek frame to inference
                        av_frame = VideoFrame.from_ndarray(bgr_frame, format="bgr24")
                        self.inference_slot.put(VideoInferItem(av_frame, self.frame_index, time.perf_counter()))
                        await self._render_and_publish(bgr_frame)
                    clock_time = time.perf_counter()
                    clock_frame = self.frame_index

                # 2. Paused handling
                if self.paused:
                    clock_time = time.perf_counter()
                    clock_frame = self.frame_index
                    if self._last_raw is not None:
                        await self._render_and_publish(self._last_raw)
                    await asyncio.sleep(PAUSE_RERENDER_INTERVAL)
                    continue

                # 3. Pacing and speed
                effective_fps = self.video_fps * max(0.1, self.speed)
                frame_interval = 1.0 / effective_fps
                target_frame = clock_frame + int((time.perf_counter() - clock_time) * effective_fps)

                # Skip frames with grab() if behind clock
                if target_frame > self.frame_index + 1:
                    while self.frame_index < target_frame - 1:
                        grabbed = await asyncio.to_thread(cap.grab)
                        if not grabbed:
                            break
                        self.frame_index += 1
                        self.dropped_frames += 1

                frame_start = time.perf_counter()
                success, bgr_frame = await asyncio.to_thread(cap.read)
                if not success:
                    if self.loop:
                        await asyncio.to_thread(cap.set, cv2.CAP_PROP_POS_FRAMES, 0)
                        self.frame_index = 0
                        clock_time = time.perf_counter()
                        clock_frame = 0
                        self.detector.reset_tracking()
                        self.processor.reset()
                        self.latest_result = None
                        self.latest_result_frame_index = None
                        continue
                    else:
                        self.finished = True
                        break

                self.frame_index += 1
                self._last_raw = bgr_frame

                # Send frame to inference slot
                av_frame = VideoFrame.from_ndarray(bgr_frame, format="bgr24")
                self.inference_slot.put(VideoInferItem(av_frame, self.frame_index, time.perf_counter()))

                # Render and publish
                await self._render_and_publish(bgr_frame)

                # Pace playback to video FPS * speed
                elapsed = time.perf_counter() - frame_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    self.dropped_frames += 1

        except asyncio.CancelledError:
            logger.info("Local video playback cancelled")
        except Exception:
            logger.exception("Local video playback error")
        finally:
            if cap:
                await asyncio.to_thread(cap.release)
            self.inference_slot.close()
            logger.info("Local video playback finished")
