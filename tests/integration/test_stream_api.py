import cv2
import numpy as np
import logging
from fastapi.testclient import TestClient

from backend.api.dependencies import get_yolo_model
from backend.domain.entities import BoundingBox, Detection, ObstacleClass
from backend.main import app


class FakeYoloModel:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        return [
            Detection(
                obstacle_class=ObstacleClass.POSTE,
                confidence=0.9,
                bounding_box=BoundingBox(
                    x_min=5.0,
                    y_min=5.0,
                    x_max=30.0,
                    y_max=30.0,
                ),
                frame_id="stream-frame",
            )
        ]


def create_sample_video(path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 64),
    )
    assert writer.isOpened()
    try:
        for _ in range(3):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    finally:
        writer.release()


def test_start_stream_notifies_websocket_and_logs_alert(
    tmp_path, caplog
) -> None:
    video_path = tmp_path / "sample.avi"
    create_sample_video(video_path)
    app.dependency_overrides[get_yolo_model] = lambda: FakeYoloModel()
    client = TestClient(app)

    try:
        with caplog.at_level(logging.INFO, logger="backend.infrastructure.notifier"):
            with client.websocket_connect("/ws/alerts") as websocket:
                response = client.post(
                    "/detect/stream/start",
                    json={"video_path": str(video_path)},
                )
                message = websocket.receive_json()
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_frames"] == 3
    assert payload["average_fps"] > 0
    assert payload["total_alerts"] == 3
    assert message["event"] == "alert"
    assert message["alert"]["risk_level"] == "ALTO"
    assert any('"event": "alert"' in record.message for record in caplog.records)
