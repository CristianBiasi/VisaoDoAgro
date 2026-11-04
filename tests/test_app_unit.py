"""Regras dos handlers: estado isolado, relógio controlado e sessão simulada."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from config import Settings
from models import Detection, SettingsPatch, Telemetry
from processing import FrameResult
from tests.support import IsolatedAppMixin


def detection(track_id=7, risk="HIGH", proximity=60):
    return Detection(track_id=track_id, object_class="tree", confidence=.9,
                     bbox=[20, 20, 80, 80], proximity=proximity, approach_rate=0,
                     position="CENTER", risk=risk, corridor_intersection=.8)


class AlertTests(IsolatedAppMixin, unittest.TestCase):
    def publish(self, now, *items):
        with patch.object(self.app.time, "perf_counter", return_value=now):
            self.app.publish_result(FrameResult(list(items), 100, 100, 10, 12, now), {})

    def test_alert_cooldown_suppresses_duplicates_but_not_escalation(self):
        self.publish(10, detection())
        self.publish(11, detection())
        self.assertEqual(self.app.state["alert_count"], 1)
        self.publish(11.5, detection(risk="CRITICAL"))
        self.assertEqual(self.app.state["alert_count"], 2)
        self.publish(13.999, detection(risk="CRITICAL"))
        self.assertEqual(self.app.state["alert_count"], 2)
        self.publish(14, detection(risk="CRITICAL"))
        self.assertEqual(self.app.state["alert_count"], 3)
        self.assertEqual([event["level"] for event in self.app.state["events"]],
                         ["CRITICAL", "CRITICAL", "HIGH"])
        self.session_factory.assert_not_called()

    def test_cooldown_is_independent_per_object_and_cleared_after_disappearance(self):
        self.publish(10, detection(1), detection(2))
        self.assertEqual(self.app.state["alert_count"], 2)
        self.publish(10.1, detection(2))
        self.publish(10.2, detection(1), detection(2))
        self.assertEqual(self.app.state["alert_count"], 3)
        self.assertEqual([event["track_id"] for event in self.app.state["events"]], [1, 2, 1])

    def test_safe_and_low_do_not_emit_alerts_medium_does_not_increment_counter(self):
        self.publish(10, detection(1, "SAFE"), detection(2, "LOW"), detection(3, "MEDIUM"))
        self.assertEqual(len(self.app.state["events"]), 1)
        self.assertEqual(self.app.state["events"][0]["level"], "MEDIUM")
        self.assertEqual(self.app.state["alert_count"], 0)

    def test_database_failure_keeps_alert_available_in_memory(self):
        self.app.state["mission_id"] = 42
        self.session_factory.side_effect = RuntimeError("database offline")
        with self.assertLogs("app", level="ERROR"):
            self.publish(10, detection())
        self.assertEqual(self.app.state["alert_count"], 1)
        self.assertEqual(self.app.state["events"][0]["track_id"], 7)
        self.assertEqual(self.app.state["detections"][0]["risk"], "HIGH")

    def test_inference_failure_removes_previous_detections(self):
        self.publish(10, detection())
        self.app.publish_error(RuntimeError("model unavailable"))
        self.assertEqual(self.app.state["detections"], [])
        self.assertEqual(self.app.state["events"][0]["level"], "WARNING")
        self.assertIsNone(self.app.snapshot()["primary_threat"])

    def test_snapshot_hides_expired_missing_or_reconfigured_results(self):
        self.app.state["detections"] = [detection().model_dump()]
        track = Mock()
        track.metrics.return_value = {"result_age_ms": 2000, "server_detection_ms": 2000}
        self.app.state["metrics"] = {"server_detection_ms": 12}
        self.app.tracks["camera"] = track
        for label, result, configuration in (
            ("expired", SimpleNamespace(received_at=8), self.app.settings),
            ("missing", None, self.app.settings),
            ("reconfigured", SimpleNamespace(received_at=10), Settings()),
        ):
            with self.subTest(reason=label), patch.object(self.app.time, "perf_counter", return_value=10):
                track.latest_result = result
                track.result_settings = configuration
                snapshot = self.app.snapshot()
                self.assertEqual(snapshot["detections"], [])
                self.assertIsNone(snapshot["primary_threat"])
                self.assertEqual(snapshot["metrics"]["objects"], 0)
                self.assertEqual(snapshot["metrics"]["server_detection_ms"], 12)

    def test_snapshot_keeps_fresh_result_and_prioritizes_severity_then_proximity(self):
        items = [detection(1, "HIGH", 99), detection(2, "CRITICAL", 80), detection(3, "CRITICAL", 90)]
        self.app.state["detections"] = [item.model_dump() for item in items]
        track = Mock(latest_result=SimpleNamespace(received_at=9), result_settings=self.app.settings)
        track.metrics.return_value = {}
        self.app.tracks["camera"] = track
        with patch.object(self.app.time, "perf_counter", return_value=10):
            snapshot = self.app.snapshot()
        self.assertEqual(snapshot["metrics"]["objects"], 3)
        self.assertEqual(snapshot["primary_threat"]["track_id"], 3)
        self.assertIn("cpu", snapshot["metrics"])
        self.assertIn("ram", snapshot["metrics"])
        self.assertIn("model", snapshot["metrics"])


class SettingsAndMissionTests(IsolatedAppMixin, unittest.IsolatedAsyncioTestCase):
    async def test_invalid_settings_preserve_entire_configuration_and_active_detections(self):
        original = self.app.settings
        before = original.model_dump()
        self.app.state["detections"] = [detection().model_dump()]
        detections = list(self.app.state["detections"])
        for changes in ({"confidence": .8, "high_threshold": 20},
                        {"medium_threshold": 58}, {"critical_threshold": 58},
                        {"performance_profile": "QUALITY", "image_size": 415},
                        {"unknown": True}):
            with self.subTest(changes=changes):
                response = await self.app.patch_settings(SettingsPatch(values=changes))
                self.assertEqual(response.status_code, 422)
                self.assertIs(self.app.settings, original)
                self.assertIs(self.app.config.settings, original)
                self.assertEqual(original.model_dump(), before)
                self.assertEqual(self.app.state["detections"], detections)
        self.session_factory.assert_not_called()
        self.assertFalse(self.app.state["events"])

    async def test_valid_settings_clear_old_detections_and_persist_explicit_override(self):
        original = self.app.settings
        self.app.state["detections"] = [detection().model_dump()]
        result = await self.app.patch_settings(SettingsPatch(values={
            "performance_profile": "QUALITY", "image_size": 512,
        }))
        self.assertEqual((result["performance_profile"], result["image_size"], result["inference_fps"]),
                         ("CUSTOM", 512, 12))
        self.assertEqual(original.inference_fps, 15)
        self.assertIs(self.app.config.settings, self.app.settings)
        self.assertEqual(self.app.state["detections"], [])
        stored = self.db.add.call_args.args[0]
        self.assertIsInstance(stored, self.app.Configuracao)
        self.assertEqual(stored.settings_json, result)
        self.db.commit.assert_called_once()

    async def test_zero_coordinates_and_sensor_values_are_available(self):
        response = await self.app.telemetry(Telemetry(latitude=0, longitude=0, heading=0))
        self.assertEqual(response, {"ok": True})
        self.assertTrue(self.app.state["gps_active"])
        self.assertTrue(self.app.state["sensors_active"])
        await self.app.telemetry(Telemetry())
        self.assertFalse(self.app.state["gps_active"])
        self.assertFalse(self.app.state["sensors_active"])
        self.session_factory.assert_not_called()

    async def test_telemetry_is_saved_with_current_mission(self):
        self.app.state["mission_id"] = 42
        payload = Telemetry(latitude=-23.5, longitude=-46.6, altitude=800, speed_kmh=12,
                            heading=90, pitch=1, roll=2)
        with patch.object(self.app.time, "time", return_value=100):
            await self.app.telemetry(payload)
        stored = self.db.add.call_args.args[0]
        self.assertIsInstance(stored, self.app.TelemetriaPonto)
        self.assertEqual((stored.missao_id, stored.timestamp, stored.lat, stored.lon),
                         (42, 100, -23.5, -46.6))
        self.assertEqual((stored.alt, stored.velocidade, stored.heading, stored.pitch, stored.roll),
                         (800, 12, 90, 1, 2))
        self.db.commit.assert_called_once()

    async def test_start_mission_resets_alerts_and_uses_persisted_id(self):
        self.app.state["alert_count"] = 5
        self.app.state["events"].append({"message": "previous mission"})
        self.app.state["last_alert"][7] = ("HIGH", 1)
        self.db.refresh.side_effect = lambda mission: setattr(mission, "id", 42)
        with patch.object(self.app.time, "time", return_value=100):
            self.assertEqual(await self.app.start_mission(), {"ok": True})
        mission = self.db.add.call_args_list[0].args[0]
        self.assertIsInstance(mission, self.app.Missao)
        self.assertEqual((mission.inicio, mission.status, mission.modelo),
                         (100, "EM_ANDAMENTO", self.app.settings.model_path))
        self.assertEqual(self.app.state["mission_id"], 42)
        self.assertEqual(self.app.state["alert_count"], 0)
        self.assertEqual(self.app.state["last_alert"], {})
        self.assertEqual([event["message"] for event in self.app.state["events"]], ["Mission started"])
        self.db.refresh.assert_called_once_with(mission)

    async def test_stop_mission_saves_duration_and_alert_count(self):
        mission = self.app.Missao(id=42, inicio=100, status="EM_ANDAMENTO")
        self.app.state["mission_id"] = 42
        self.app.state["alert_count"] = 3
        self.db.query.return_value.filter.return_value.first.return_value = mission
        with patch.object(self.app.time, "time", return_value=145):
            self.assertEqual(await self.app.stop_mission(), {"ok": True})
        self.assertEqual((mission.fim, mission.duracao, mission.status, mission.alertas),
                         (145, 45, "CONCLUIDA", 3))
        self.assertIsNone(self.app.state["mission_id"])
        self.db.commit.assert_called_once()

    async def test_view_mode_does_not_mutate_settings_or_emit_events(self):
        original = self.app.settings
        self.app.state["events"].clear()
        res = await self.app.set_view_mode(self.app.ViewModePayload(mode="RAW"))
        self.assertEqual(res, {"mode": "RAW"})
        self.assertEqual(self.app.state["view_mode"], "RAW")
        self.assertEqual(self.app.snapshot()["view_mode"], "RAW")
        self.assertIs(self.app.settings, original)
        self.assertEqual(list(self.app.state["events"]), [])

    async def test_demo_videos_listing_and_rejection(self):
        demos = await self.app.list_demo_videos()
        self.assertIsInstance(demos, list)

        for bad_name in ("../app.py", "nonexistent.mp4", "evil/path.mp4", "test.txt"):
            res = await self.app.start_demo_video(self.app.DemoVideoRequest(name=bad_name))
            self.assertEqual(res.status_code, 400)
