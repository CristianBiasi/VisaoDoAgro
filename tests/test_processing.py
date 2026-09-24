"""Contrato do processamento com detector simulado e regras reais."""
import unittest
from unittest.mock import Mock

import numpy as np

from config import Settings, updated_settings
from detector import ObjectDetection
from processing import DetectionProcessor


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.frame = Mock()
        self.frame.to_ndarray.return_value = self.image
        self.detector = Mock(generation=1, last_inference=12.5)
        self.detector.detect.return_value = [ObjectDetection(7, "tree", .9, [20, 20, 80, 80], True)]
        self.processor = DetectionProcessor(self.detector)
        self.settings = Settings(confirmation_frames=2, proximity_smoothing=1)

    def process(self, timestamp, settings=None):
        return self.processor.process(self.frame, settings or self.settings, timestamp)

    def confirm_threat(self):
        self.assertEqual(self.process(1).detections[0].risk, "SAFE")
        self.assertEqual(self.process(2).detections[0].risk, "CRITICAL")

    def test_tracked_threat_is_confirmed_and_preserves_detection_contract(self):
        self.confirm_threat()
        result = self.process(3)
        item = result.detections[0]
        self.assertEqual((result.width, result.height, result.inference_ms, result.received_at),
                         (100, 100, 12.5, 3))
        self.assertEqual((item.track_id, item.object_class, item.confidence), (7, "tree", .9))
        self.assertEqual((item.position, item.proximity, item.approach_rate), ("CENTER", 90, 0))
        self.assertEqual(item.bbox, [20, 20, 80, 80])
        self.frame.to_ndarray.assert_called_with(format="bgr24")
        self.assertIs(self.detector.detect.call_args.args[0], self.image)

    def test_disappearing_object_must_be_confirmed_again(self):
        self.confirm_threat()
        objects = self.detector.detect.return_value
        self.detector.detect.return_value = []
        self.assertEqual(self.process(3).detections, [])
        self.detector.detect.return_value = objects
        result = self.process(4).detections[0]
        self.assertEqual((result.risk, result.approach_rate), ("SAFE", 0))
        self.assertEqual(self.process(5).detections[0].risk, "CRITICAL")

    def test_model_reload_discards_confirmation_and_proximity_history(self):
        self.confirm_threat()
        self.detector.generation = 2
        self.detector.detect.return_value = [ObjectDetection(7, "tree", .9, [10, 10, 90, 90], True)]
        item = self.process(3).detections[0]
        self.assertEqual((item.risk, item.approach_rate), ("SAFE", 0))
        self.assertEqual(self.process(4).detections[0].risk, "CRITICAL")

    def test_safety_settings_change_requires_new_confirmation(self):
        self.confirm_threat()
        changed = updated_settings(self.settings, {"corridor_width": .8})
        self.assertEqual(self.process(3, changed).detections[0].risk, "SAFE")
        self.assertEqual(self.process(4, changed).detections[0].risk, "CRITICAL")

    def test_interface_settings_do_not_reset_confirmed_threat(self):
        self.confirm_threat()
        changed = updated_settings(self.settings, {"sound_enabled": False})
        self.assertEqual(self.process(3, changed).detections[0].risk, "CRITICAL")

    def test_untracked_detections_do_not_share_history_even_with_tracking_enabled(self):
        for timestamp, bbox in enumerate(([20, 20, 80, 80], [10, 10, 90, 90]), 1):
            self.detector.detect.return_value = [ObjectDetection(-1, "tree", .9, bbox, False)]
            item = self.process(timestamp).detections[0]
            self.assertEqual((item.risk, item.approach_rate), ("CRITICAL", 0))

    def test_position_boundaries(self):
        for center, expected in ((37, "LEFT"), (38, "CENTER"), (62, "CENTER"), (63, "RIGHT")):
            with self.subTest(center=center):
                self.detector.detect.return_value = [
                    ObjectDetection(7, "tree", .9, [center - 5, 10, center + 5, 20], True),
                ]
                self.assertEqual(self.process(1).detections[0].position, expected)
