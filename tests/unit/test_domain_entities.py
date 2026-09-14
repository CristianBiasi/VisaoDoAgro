from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from backend.domain.entities import (
    Alert,
    BoundingBox,
    Detection,
    ObstacleClass,
    RiskLevel,
)


@pytest.fixture
def valid_bounding_box() -> BoundingBox:
    return BoundingBox(x_min=10.0, y_min=20.0, x_max=100.0, y_max=200.0)


@pytest.fixture
def valid_detection(valid_bounding_box: BoundingBox) -> Detection:
    return Detection(
        obstacle_class=ObstacleClass.ARVORE,
        confidence=0.95,
        bounding_box=valid_bounding_box,
        frame_id="frame-001",
    )


def test_obstacle_class_accepts_valid_value() -> None:
    assert ObstacleClass("ARVORE") is ObstacleClass.ARVORE


def test_obstacle_class_rejects_invalid_value() -> None:
    with pytest.raises(ValueError):
        ObstacleClass("ANIMAL")


def test_bounding_box_accepts_valid_coordinates() -> None:
    box = BoundingBox(x_min=0.0, y_min=1.0, x_max=10.0, y_max=20.0)

    assert box.x_min == 0.0
    assert box.y_max == 20.0


def test_bounding_box_rejects_inverted_coordinates() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(x_min=10.0, y_min=1.0, x_max=0.0, y_max=20.0)


def test_detection_accepts_valid_values(valid_detection: Detection) -> None:
    assert valid_detection.obstacle_class is ObstacleClass.ARVORE
    assert valid_detection.confidence == 0.95


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_detection_rejects_confidence_outside_range(
    valid_bounding_box: BoundingBox, confidence: float
) -> None:
    with pytest.raises(ValidationError):
        Detection(
            obstacle_class=ObstacleClass.POSTE,
            confidence=confidence,
            bounding_box=valid_bounding_box,
            frame_id="frame-002",
        )


def test_risk_level_accepts_valid_value() -> None:
    assert RiskLevel("ALTO") is RiskLevel.ALTO


def test_risk_level_rejects_invalid_value() -> None:
    with pytest.raises(ValueError):
        RiskLevel("CRITICO")


def test_alert_accepts_valid_values(valid_detection: Detection) -> None:
    timestamp = datetime.now(timezone.utc)
    alert = Alert(
        detection=valid_detection,
        risk_level=RiskLevel.MEDIO,
        message="Obstáculo identificado",
        timestamp=timestamp,
    )

    assert alert.detection == valid_detection
    assert alert.timestamp == timestamp


def test_alert_rejects_invalid_detection(valid_bounding_box: BoundingBox) -> None:
    with pytest.raises(ValidationError):
        Alert(
            detection={
                "obstacle_class": "AVE",
                "confidence": 1.5,
                "bounding_box": valid_bounding_box,
                "frame_id": "frame-003",
            },
            risk_level=RiskLevel.BAIXO,
            message="Obstáculo identificado",
            timestamp=datetime.now(timezone.utc),
        )


def test_alert_rejects_empty_message(valid_detection: Detection) -> None:
    with pytest.raises(ValidationError):
        Alert(
            detection=valid_detection,
            risk_level=RiskLevel.BAIXO,
            message="",
            timestamp=datetime.now(timezone.utc),
        )
