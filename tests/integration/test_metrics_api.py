import json

from fastapi.testclient import TestClient

from backend.api.dependencies import get_detection_repository
from backend.domain.entities import BoundingBox, Detection, ObstacleClass
from backend.main import app
from backend.repositories.detection_repository import DetectionRepository


def make_detection(frame_id: str, obstacle_class: ObstacleClass, offset: float = 0) -> Detection:
    return Detection(
        obstacle_class=obstacle_class,
        confidence=0.9,
        bounding_box=BoundingBox(
            x_min=offset,
            y_min=offset,
            x_max=offset + 10,
            y_max=offset + 10,
        ),
        frame_id=frame_id,
    )


def test_metrics_report_matches_manual_counts(tmp_path) -> None:
    database = DetectionRepository(tmp_path / "metrics.db")
    database.save_results(
        [
            make_detection("frame-1", ObstacleClass.POSTE),
            make_detection("frame-3", ObstacleClass.AVE),
            make_detection("frame-4", ObstacleClass.ARVORE),
        ],
        [],
    )
    annotation_path = tmp_path / "ground_truth.json"
    annotation_path.write_text(
        json.dumps(
            [
                {
                    "frame_id": "frame-1",
                    "obstacle_class": "POSTE",
                    "bounding_box": {
                        "x_min": 0,
                        "y_min": 0,
                        "x_max": 10,
                        "y_max": 10,
                    },
                },
                {
                    "frame_id": "frame-2",
                    "obstacle_class": "AVE",
                    "bounding_box": {
                        "x_min": 0,
                        "y_min": 0,
                        "x_max": 10,
                        "y_max": 10,
                    },
                },
            ]
        ),
        encoding="utf-8",
    )
    app.dependency_overrides[get_detection_repository] = lambda: database
    client = TestClient(app)

    try:
        response = client.get(
            "/metrics/report",
            params={"annotation_path": str(annotation_path)},
        )
    finally:
        app.dependency_overrides.clear()
        database.close()

    assert response.status_code == 200
    payload = response.json()
    assert payload["true_positives"] == 1
    assert payload["false_positives"] == 2
    assert payload["false_negatives"] == 1
    assert payload["precision"] == 1 / 3
    assert payload["recall"] == 1 / 2
    assert payload["map_simplified"] == 1 / 3
