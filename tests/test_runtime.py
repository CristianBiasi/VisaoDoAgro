import asyncio
import time
import unittest
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from av import VideoFrame
from pydantic import ValidationError
from aiortc.mediastreams import MediaStreamError

from config import Settings, PERFORMANCE_PROFILES, PROXIMITY_PRESETS, updated_settings
from detector import ObjectDetection, YoloDetector, select_device
from processing import DetectionProcessor, FrameResult
from proximity import ProximityEstimator
from risk import CollisionRiskEngine, Risk
from video_pipeline import LatestFrame, ProcessedTrack


def frame(value=0, pts=0):
    result = VideoFrame.from_ndarray(np.full((64, 96, 3), value, dtype=np.uint8), format="bgr24")
    result.pts = pts
    result.time_base = Fraction(1, 30)
    return result


class SettingsTests(unittest.TestCase):
    def test_profiles_and_explicit_override(self):
        for name, values in PERFORMANCE_PROFILES.items():
            configuration = updated_settings(Settings(), {"performance_profile": name})
            for field, value in values.items():
                self.assertEqual(getattr(configuration, field), value)
        custom = updated_settings(Settings(), {"performance_profile": "QUALITY", "image_size": 512})
        self.assertEqual(custom.image_size, 512)
        self.assertEqual(custom.performance_profile, "CUSTOM")

    def test_proximity_presets_are_monotonic(self):
        scores = []
        for name in ("NEAR", "NORMAL", "FAR"):
            configuration = updated_settings(Settings(), {"proximity_preset": name})
            self.assertLess(configuration.medium_threshold, configuration.high_threshold)
            self.assertLess(configuration.high_threshold, configuration.critical_threshold)
            estimator = ProximityEstimator()
            scores.append(estimator.update(1, [0, 0, 20, 20], 100, 100, 1,
                                           configuration.proximity_scale).score)
        self.assertLess(scores[0], scores[1])
        self.assertLess(scores[1], scores[2])
        self.assertEqual(set(PROXIMITY_PRESETS), {"NEAR", "NORMAL", "FAR"})

    def test_invalid_settings_leave_original_unchanged(self):
        original = Settings()
        for changes in ({"inference_fps": 0}, {"confidence": 2}, {"high_threshold": 20},
                        {"image_size": 415}, {"device": "garbage"}, {"unknown": True},
                        {"model_path": ""}, {"model_path": "best.onnx"},
                        {"performance_profile": "FAST"}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                updated_settings(original, changes)
        self.assertEqual(original.inference_fps, 15)

    def test_presets_do_not_reset_unrelated_advanced_fields(self):
        current = Settings(corridor_width=0.8, tracking_enabled=False)
        updated = updated_settings(current, {"proximity_preset": "FAR"})
        self.assertEqual(updated.corridor_width, 0.8)
        self.assertFalse(updated.tracking_enabled)
        self.assertEqual(Settings(performance_profile="QUALITY").image_size, 640)


class SafetyTests(unittest.TestCase):
    def test_history_is_removed_and_score_is_bounded(self):
        estimator = ProximityEstimator(smoothing=1)
        self.assertEqual(estimator.update(1, [0, 0, 999, 999], 100, 100, 1).score, 100)
        self.assertEqual(estimator.update(2, [20, 20, 0, 0], 100, 100, 1).score, 0)
        estimator.forget_missing(set())
        self.assertEqual(len(estimator.history), 0)
        self.assertEqual(len(estimator.smoothed), 0)

    def test_different_candidates_do_not_share_confirmation(self):
        engine = CollisionRiskEngine()
        self.assertEqual(engine.update(1, Risk.HIGH, 2), Risk.SAFE)
        self.assertEqual(engine.update(1, Risk.LOW, 2), Risk.SAFE)
        self.assertEqual(engine.update(1, Risk.HIGH, 2), Risk.SAFE)
        self.assertEqual(engine.update(1, Risk.HIGH, 2), Risk.HIGH)
        self.assertEqual(engine.update(1, Risk.LOW, 2), Risk.HIGH)
        self.assertEqual(engine.update(1, Risk.LOW, 2), Risk.HIGH)
        self.assertEqual(engine.update(1, Risk.LOW, 2), Risk.LOW)
        engine.forget_missing(set())
        self.assertFalse(engine.pending)

    def test_predict_mode_does_not_reuse_approach_history(self):
        detector = Mock()
        detector.last_inference = 1
        detector.detect.side_effect = [
            [ObjectDetection(-1, "tree", .9, [0, 0, 10, 10], False)],
            [ObjectDetection(-1, "tree", .9, [0, 0, 60, 60], False)],
        ]
        processor = DetectionProcessor(detector)
        configuration = Settings(tracking_enabled=False)
        processor.process(frame(), configuration, 1)
        result = processor.process(frame(), configuration, 2)
        self.assertEqual(result.detections[0].approach_rate, 0)
        self.assertNotEqual(result.detections[0].risk, "SAFE")


class DetectorTests(unittest.TestCase):
    def make_detector(self, rows=None, names=None, device="cpu"):
        data = Mock()
        data.cpu.return_value.tolist.return_value = rows or [[1, 2, 30, 40, 7, .9, 0]]
        model = Mock()
        model.names = names or {0: "tree", 1: "wire"}
        model.predictor = None
        model.track.return_value = [SimpleNamespace(boxes=SimpleNamespace(data=data))]
        model.predict.return_value = model.track.return_value
        factory = Mock(return_value=model)
        detector = YoloDetector(factory, lambda requested: device)
        return detector, model, factory, data

    def test_custom_names_and_single_cpu_transfer(self):
        detector, model, factory, data = self.make_detector()
        with patch("detector.Path.is_file", return_value=True):
            result = detector.detect(np.zeros((64, 64, 3)), Settings(model_path="models/best.pt", monitored_classes=["tree"]))
        self.assertEqual(result[0].object_class, "tree")
        self.assertEqual(result[0].track_id, 7)
        data.cpu.assert_called_once()
        self.assertEqual(model.track.call_args.kwargs["classes"], [0])

    def test_missing_classes_warn_without_detecting_everything(self):
        detector, model, _, _ = self.make_detector()
        with patch("detector.Path.is_file", return_value=True), self.assertLogs("detector", level="WARNING"):
            self.assertEqual(detector.detect(np.zeros((64, 64, 3)), Settings(monitored_classes=["person"])), [])
        model.track.assert_not_called()
        self.assertEqual(detector.missing_classes, ["person"])

    def test_empty_classes_select_all_and_predict_ids_are_frame_local(self):
        detector, model, _, _ = self.make_detector(rows=[[1, 2, 3, 4, .8, 1]])
        with patch("detector.Path.is_file", return_value=True):
            result = detector.detect(np.zeros((64, 64, 3)), Settings(monitored_classes=[], tracking_enabled=False))
        self.assertIsNone(model.predict.call_args.kwargs["classes"])
        self.assertEqual(result[0].track_id, -1)
        self.assertFalse(result[0].tracked)

    def test_model_device_and_precision_reload_but_confidence_does_not(self):
        detector, model, factory, _ = self.make_detector()
        settings = Settings(monitored_classes=[])
        with patch("detector.Path.is_file", return_value=True):
            detector.detect(np.zeros((64, 64, 3)), settings)
            detector.detect(np.zeros((64, 64, 3)), updated_settings(settings, {"confidence": .6}))
            self.assertEqual(factory.call_count, 1)
            for changes in ({"model_path": "models/best.pt"}, {"device": "cpu"}, {"half_precision": True}):
                settings = updated_settings(settings, changes)
                detector.detect(np.zeros((64, 64, 3)), settings)
        self.assertEqual(factory.call_count, 4)

    def test_disabling_tracking_reloads_to_remove_callbacks(self):
        detector, model, factory, _ = self.make_detector()
        settings = Settings(monitored_classes=[])
        with patch("detector.Path.is_file", return_value=True):
            detector.detect(np.zeros((64, 64, 3)), settings)
            detector.detect(np.zeros((64, 64, 3)), updated_settings(settings, {"tracking_enabled": False}))
        self.assertEqual(factory.call_count, 2)
        model.predict.assert_called_once()

    def test_failed_load_does_not_commit_a_new_model(self):
        detector, _, factory, _ = self.make_detector()
        with patch("detector.Path.is_file", return_value=True):
            detector.detect(np.zeros((64, 64, 3)), Settings(monitored_classes=[]))
        with patch("detector.Path.is_file", return_value=False), self.assertRaises(FileNotFoundError):
            detector.detect(np.zeros((64, 64, 3)), Settings(model_path="models/missing.pt"))
        self.assertEqual(detector.loaded_model, "yolo11n.pt")
        self.assertEqual(factory.call_count, 1)

    def test_device_selection_falls_back_without_cuda(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        self.assertEqual(select_device("auto", torch), "cpu")
        with self.assertLogs("detector", level="WARNING"):
            self.assertEqual(select_device("cuda:0", torch), "cpu")
        cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1))
        self.assertEqual(select_device("auto", cuda), "cuda:0")
        self.assertEqual(select_device("cpu", cuda), "cpu")

    def test_cpu_disables_half(self):
        detector, model, _, _ = self.make_detector()
        with patch("detector.Path.is_file", return_value=True):
            detector.detect(np.zeros((64, 64, 3)), Settings(monitored_classes=[], half_precision=True))
        self.assertFalse(model.track.call_args.kwargs["half"])

    def test_cuda_failure_retries_cpu(self):
        detector, model, _, _ = self.make_detector(device="cuda:0")
        detector._select_device = lambda requested: "cpu" if requested == "cpu" else "cuda:0"
        success = model.track.return_value
        model.track.side_effect = [RuntimeError("CUDA failure"), success]
        with patch("detector.Path.is_file", return_value=True), self.assertLogs("detector", level="ERROR"):
            result = detector.detect(np.zeros((64, 64, 3)), Settings(monitored_classes=[], half_precision=True))
        self.assertTrue(result)
        self.assertEqual(detector.device, "cpu")
        self.assertFalse(model.track.call_args.kwargs["half"])


class LatestFrameTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_newest_frame_survives(self):
        slot = LatestFrame()
        for value in range(100):
            slot.put(value)
        self.assertEqual(await slot.get(), 99)
        self.assertEqual(slot.dropped, 99)
        self.assertIsNone(slot.value)

    async def test_close_wakes_waiter(self):
        slot = LatestFrame()
        waiter = asyncio.create_task(slot.get())
        await asyncio.sleep(0)
        slot.close()
        with self.assertRaises(MediaStreamError):
            await waiter


class FakeSource:
    def __init__(self):
        self.queue = asyncio.Queue()

    async def recv(self):
        result = await self.queue.get()
        if result is None:
            raise MediaStreamError
        return result

    def stop(self):
        self.queue.put_nowait(None)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_inference_drops_backlog_and_does_not_block_loop(self):
        source = FakeSource()
        detector = Mock(device="cpu", half=False)
        settings = Settings(inference_fps=60)
        completed = []
        track = ProcessedTrack(source, detector, lambda: settings,
                               lambda result, metrics: completed.append(result), lambda exc: None, set())
        seen = []

        def slow_process(video_frame, configuration, received_at):
            seen.append(video_frame.pts)
            time.sleep(.08)
            return FrameResult([], 96, 64, 80, 80, received_at)

        track.processor.process = slow_process
        track._start()
        source.queue.put_nowait(frame(pts=0))
        await asyncio.sleep(.02)
        for index in range(1, 21):
            source.queue.put_nowait(frame(pts=index))
        await asyncio.sleep(.02)
        self.assertGreater(track.inference_slot.dropped, 0)
        self.assertEqual(track.inference_slot.value.frame.pts, 20)
        await asyncio.sleep(.20)
        self.assertEqual(seen, [0, 20])
        await track.aclose()
        self.assertTrue(track._infer_task.done())

    async def test_video_uses_current_frame_and_preserves_timestamps(self):
        source = FakeSource()
        detector = Mock(device="cpu", half=False)
        settings = Settings()
        track = ProcessedTrack(source, detector, lambda: settings, lambda *args: None, lambda exc: None, set())
        track.processor.process = lambda video_frame, cfg, received: FrameResult([], 96, 64, 0, 0, received)
        with patch("video_pipeline.cv2.imencode") as encode:
            source.queue.put_nowait(frame(20, 1))
            first = await track.recv()
            source.queue.put_nowait(frame(80, 2))
            second = await track.recv()
            self.assertEqual((first.pts, second.pts), (1, 2))
            self.assertEqual(second.time_base, Fraction(1, 30))
            self.assertGreater(second.to_ndarray().mean(), first.to_ndarray().mean())
            encode.assert_not_called()
        await track.aclose()

    async def test_overlay_does_not_modify_source_frame(self):
        from models import Detection
        source = FakeSource()
        detector = Mock(device="cpu", half=False)
        settings = Settings()
        track = ProcessedTrack(source, detector, lambda: settings, lambda *args: None, lambda exc: None, set())
        original = frame()
        before = original.to_ndarray(format="bgr24").copy()
        detection = Detection(track_id=1, object_class="tree", confidence=.9,
                              bbox=[10, 10, 40, 40], proximity=30, approach_rate=0,
                              position="CENTER", risk="MEDIUM", corridor_intersection=.5)
        from video_pipeline import ReceivedFrame
        result = FrameResult([detection], 96, 64, 0, 0, time.perf_counter())
        output, jpeg, _, _ = track._render(ReceivedFrame(original, time.perf_counter()), settings, result, True)
        self.assertTrue(np.array_equal(original.to_ndarray(format="bgr24"), before))
        self.assertIsNotNone(jpeg)
        self.assertFalse(np.array_equal(output.to_ndarray(format="bgr24"), before))
        await track.aclose()

    async def test_settings_change_during_inference_discards_old_result(self):
        source = FakeSource()
        detector = Mock(device="cpu", half=False)
        current = [Settings()]
        published = []
        track = ProcessedTrack(source, detector, lambda: current[0],
                               lambda *args: published.append(args), lambda exc: None, set())

        def slow_process(video_frame, cfg, received):
            time.sleep(.08)
            return FrameResult([], 96, 64, 80, 80, received)

        track.processor.process = slow_process
        track._start()
        source.queue.put_nowait(frame())
        await asyncio.sleep(.02)
        current[0] = updated_settings(current[0], {"confidence": .8})
        await asyncio.sleep(.1)
        self.assertFalse(published)
        await track.aclose()


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import app
        self.app = app
        self.original = app.settings

    async def asyncTearDown(self):
        self.app.settings = self.original
        self.app.config.settings = self.original
        self.app.state["detections"] = []

    async def test_patch_rejects_invalid_and_applies_valid_to_runtime(self):
        from models import SettingsPatch
        response = await self.app.patch_settings(SettingsPatch(values={"inference_fps": 0}))
        self.assertEqual(response.status_code, 422)
        self.assertIs(self.app.settings, self.original)
        result = await self.app.patch_settings(SettingsPatch(values={"performance_profile": "QUALITY"}))
        self.assertEqual(result["image_size"], 640)
        self.assertIs(self.app.settings, self.app.config.settings)

    async def test_primary_threat_uses_severity_not_alphabet(self):
        self.app.state["detections"] = [
            {"risk": "SAFE", "proximity": 99}, {"risk": "CRITICAL", "proximity": 80},
        ]
        self.assertEqual(self.app.snapshot()["primary_threat"]["risk"], "CRITICAL")
