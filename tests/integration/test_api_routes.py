import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.api.dependencies import get_yolo_model
from backend.domain.entities import BoundingBox, Detection, ObstacleClass
from backend.main import app


class FakeYoloModel:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        return [
            Detection(
                obstacle_class=ObstacleClass.FIO_ELETRICO,
                confidence=0.92,
                bounding_box=BoundingBox(
                    x_min=10.0,
                    y_min=20.0,
                    x_max=50.0,
                    y_max=60.0,
                ),
                frame_id="api-frame-1",
            )
        ]


def test_detect_image_returns_detections_and_alerts() -> None:
    app.dependency_overrides[get_yolo_model] = lambda: FakeYoloModel()
    client = TestClient(app)
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    _, encoded_image = cv2.imencode(".jpg", image)

    try:
        response = client.post(
            "/detect/image",
            files={
                "file": (
                    "example.jpg",
                    encoded_image.tobytes(),
                    "image/jpeg",
                )
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["detection_count"] == 1
    assert payload["alert_count"] == 1
    assert payload["detections"][0]["obstacle_class"] == "FIO_ELETRICO"
    assert payload["alerts"][0]["risk_level"] == "ALTO"
    assert payload["formatted_alerts"]
