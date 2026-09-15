"""Bounded latest-frame reception and one serial inference worker."""
import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass

import cv2
from aiortc import VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame

from overlay import draw_overlay
from processing import DetectionProcessor

logger = logging.getLogger(__name__)
JPEG_FPS = 12
MAX_OVERLAY_AGE = 1.0


class LatestFrame:
    """A single replaceable slot. Access only from the event loop."""

    def __init__(self):
        self.value = None
        self.event = asyncio.Event()
        self.closed = False
        self.dropped = 0

    def put(self, value):
        if self.closed:
            return
        if self.value is not None:
            self.dropped += 1
        self.value = value
        self.event.set()

    async def get(self):
        await self.event.wait()
        if self.closed:
            raise MediaStreamError
        value = self.value
        self.value = None
        self.event.clear()
        return value

    def close(self):
        self.closed = True
        self.value = None
        self.event.set()


class RateMeter:
    """Observed completions per second over a rolling window."""

    def __init__(self):
        self.started = time.perf_counter()
        self.times = deque()

    def tick(self, now):
        self.times.append(now)

    def value(self, now):
        while self.times and now - self.times[0] > 2:
            self.times.popleft()
        return len(self.times) / max(0.001, min(2, now - self.started))


@dataclass
class ReceivedFrame:
    frame: object
    received_at: float


class ProcessedTrack(VideoStreamTrack):
    def __init__(self, source, detector, get_settings, on_result, on_error, video_clients):
        super().__init__()
        self.source = source
        self.detector = detector
        self.processor = DetectionProcessor(detector)
        self.get_settings = get_settings
        self.on_result = on_result
        self.on_error = on_error
        self.video_clients = video_clients
        self.capture_slot = LatestFrame()
        self.inference_slot = LatestFrame()
        self.latest_result = None
        self.result_settings = None
        self.ai_rate = RateMeter()
        self.video_rate = RateMeter()
        self.last_jpeg = 0.0
        self.overlay_ms = 0.0
        self.jpeg_ms = 0.0
        self._receive_task = None
        self._infer_task = None
        self._render_task = None
        self._error_at = 0.0

    def _start(self):
        if self._receive_task is None:
            self.detector.reset_tracking()
            self._receive_task = asyncio.create_task(self._receive())
            self._infer_task = asyncio.create_task(self._infer())

    async def _receive(self):
        try:
            while self.readyState == "live":
                frame = await self.source.recv()
                item = ReceivedFrame(frame, time.perf_counter())
                self.capture_slot.put(item)
                self.inference_slot.put(item)
        except MediaStreamError:
            pass
        except Exception:
            logger.exception("Video reception failed")
        finally:
            self.capture_slot.close()
            self.inference_slot.close()

    async def _infer(self):
        next_inference = 0.0
        try:
            while self.readyState == "live":
                # Wait before taking the slot so a delayed worker always takes the newest frame.
                await asyncio.sleep(max(0, next_inference - time.perf_counter()))
                received = await self.inference_slot.get()
                configuration = self.get_settings()
                started = time.perf_counter()
                try:
                    result = await asyncio.to_thread(
                        self.processor.process, received.frame, configuration, received.received_at,
                    )
                    if self.readyState != "live":
                        break
                    # An update during inference invalidates the old result.
                    if configuration is not self.get_settings():
                        continue
                    self.latest_result = result
                    self.result_settings = configuration
                    now = time.perf_counter()
                    self.ai_rate.tick(now)
                    self.on_result(result, self.metrics(now))
                except Exception as exc:
                    self.latest_result = None
                    self.result_settings = None
                    if time.perf_counter() - self._error_at >= 5:
                        logger.exception("Inference failed")
                        self.on_error(exc)
                        self._error_at = time.perf_counter()
                    # Missing/corrupt models must not create a hot retry loop.
                    next_inference = time.perf_counter() + 1
                    continue
                next_inference = started + 1 / configuration.inference_fps
        except MediaStreamError:
            pass

    def metrics(self, now=None):
        now = time.perf_counter() if now is None else now
        result = self.latest_result
        return {
            "ai_fps": round(self.ai_rate.value(now), 1),
            "video_fps": round(self.video_rate.value(now), 1),
            "server_detection_ms": round((now - result.received_at) * 1000, 1) if result else None,
            "result_age_ms": round((now - result.received_at) * 1000, 1) if result else None,
            "inference_ms": round(result.inference_ms, 1) if result else 0,
            "processing_ms": round(result.processing_ms, 1) if result else 0,
            "overlay_ms": round(self.overlay_ms, 1),
            "jpeg_ms": round(self.jpeg_ms, 1),
            "dropped_frames": self.inference_slot.dropped,
            "device": self.detector.device,
            "backend": "Ultralytics / PyTorch",
            "half_precision": self.detector.half,
        }

    def _render(self, received, configuration, result, send_jpeg):
        started = time.perf_counter()
        image = received.frame.to_ndarray(format="bgr24")
        if result and configuration.overlay_enabled:
            # A BGR AV frame may share memory with to_ndarray; never paint into inference input.
            image = image.copy()
            height, width = image.shape[:2]
            detections = result.detections
            if (width, height) != (result.width, result.height):
                detections = [item.model_copy(update={"bbox": [
                    value * (width / result.width if index % 2 == 0 else height / result.height)
                    for index, value in enumerate(item.bbox)
                ]}) for item in detections]
            draw_overlay(image, detections, configuration)
        overlay_ms = (time.perf_counter() - started) * 1000
        output = VideoFrame.from_ndarray(image, format="bgr24")
        output.pts = received.frame.pts
        output.time_base = received.frame.time_base
        encoded_bytes = None
        jpeg_ms = 0.0
        if send_jpeg:
            started = time.perf_counter()
            success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if success:
                encoded_bytes = encoded.tobytes()
            jpeg_ms = (time.perf_counter() - started) * 1000
        return output, encoded_bytes, overlay_ms, jpeg_ms

    async def recv(self):
        self._start()
        received = await self.capture_slot.get()
        configuration = self.get_settings()
        result = self.latest_result
        now = time.perf_counter()
        if configuration is not self.result_settings or (result and now - result.received_at > MAX_OVERLAY_AGE):
            result = None
        send_jpeg = bool(self.video_clients) and now - self.last_jpeg >= 1 / JPEG_FPS
        self._render_task = asyncio.create_task(asyncio.to_thread(
            self._render, received, configuration, result, send_jpeg,
        ))
        output, jpeg, self.overlay_ms, self.jpeg_ms = await asyncio.shield(self._render_task)
        self.video_rate.tick(time.perf_counter())
        if jpeg:
            # Each dashboard has a single pending JPEG; slow clients never block WebRTC.
            for slot in tuple(self.video_clients):
                slot.put(jpeg)
            self.last_jpeg = now
        return output

    def stop(self):
        super().stop()
        self.source.stop()
        self.capture_slot.close()
        self.inference_slot.close()
        if self._receive_task:
            self._receive_task.cancel()

    async def aclose(self):
        self.stop()
        # Do not cancel to_thread inference: cancellation would leave the model running.
        tasks = [task for task in (self._receive_task, self._infer_task, self._render_task) if task]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
