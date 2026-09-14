import json

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.api.dependencies import get_detection_repository, get_yolo_model
from backend.domain.entities import BoundingBox, Detection, ObstacleClass
from backend.main import app
from backend.repositories.detection_repository import DetectionRepository


class FakeYoloModel:
    def __init__(self) -> None:
        self.frame_number = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.frame_number += 1
        return [
            Detection(
                obstacle_class=ObstacleClass.POSTE,
                confidence=0.95,
                bounding_box=BoundingBox(
                    x_min=5.0,
                    y_min=5.0,
                    x_max=30.0,
                    y_max=30.0,
                ),
                frame_id=f"frame-{self.frame_number}",
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


def test_complete_video_alert_persistence_and_metrics_flow(tmp_path) -> None:
    video_path = tmp_path / "demo.avi"
    annotation_path = tmp_path / "ground_truth.json"
    database = DetectionRepository(tmp_path / "demo.db")
    create_sample_video(video_path)
    annotation_path.write_text(
        json.dumps(
            [
                {
                    "frame_id": f"frame-{index}",
                    "obstacle_class": "POSTE",
                    "bounding_box": {
                        "x_min": 5,
                        "y_min": 5,
                        "x_max": 30,
                        "y_max": 30,
                    },
                }
                for index in range(1, 4)
            ]
        ),
        encoding="utf-8",
    )
    app.dependency_overrides[get_yolo_model] = lambda: FakeYoloModel()
    app.dependency_overrides[get_detection_repository] = lambda: database
    client = TestClient(app)

    try:
        with client.websocket_connect("/ws/alerts") as websocket:
            stream_response = client.post(
                "/detect/stream/start",
                json={"video_path": str(video_path)},
            )
            websocket_messages = [websocket.receive_json() for _ in range(3)]

        metrics_response = client.get(
            "/metrics/report",
            params={"annotation_path": str(annotation_path)},
        )
        persisted_alerts = database.count_alerts()
        persisted_detections = len(database.get_detections())
    finally:
        app.dependency_overrides.clear()
        database.close()

    assert stream_response.status_code == 200
    assert stream_response.json()["total_frames"] == 3
    assert stream_response.json()["total_alerts"] == 3
    assert len(websocket_messages) == 3
    assert all(message["alert"]["risk_level"] == "ALTO" for message in websocket_messages)
    assert persisted_alerts == 3
    assert persisted_detections == 3

    assert metrics_response.status_code == 200
    metrics = metrics_response.json()
    assert metrics["true_positives"] == 3
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
