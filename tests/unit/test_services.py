from datetime import datetime, timezone
from unittest.mock import Mock

import numpy as np

from backend.domain.entities import (
    Alert,
    BoundingBox,
    Detection,
    ObstacleClass,
    RiskLevel,
)
from backend.infrastructure.yolo_model import YoloModel
from backend.services.alert_service import AlertService
from backend.services.detection_service import DetectionService
from backend.services.risk_analysis_service import RiskAnalysisService


def make_detection(
    obstacle_class: ObstacleClass,
    *,
    area: float = 100.0,
    frame_id: str = "frame-1",
) -> Detection:
    width = area**0.5
    return Detection(
        obstacle_class=obstacle_class,
        confidence=0.9,
        bounding_box=BoundingBox(
            x_min=0.0,
            y_min=0.0,
            x_max=width,
            y_max=width,
        ),
        frame_id=frame_id,
    )


def test_detection_service_returns_empty_list_without_detections() -> None:
    model = Mock(spec=YoloModel)
    model.detect.return_value = []
    service = DetectionService(model)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    result = service.detect(frame)

    assert result == []
    model.detect.assert_called_once_with(frame)


def test_detection_service_delegates_to_injected_model() -> None:
    detection = make_detection(ObstacleClass.ARVORE)
    model = Mock(spec=YoloModel)
    model.detect.return_value = [detection]
    service = DetectionService(model)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    result = service.detect(frame)

    assert result == [detection]
    model.detect.assert_called_once_with(frame)


def test_risk_analysis_returns_no_alerts_for_empty_detections() -> None:
    assert RiskAnalysisService().analyze([]) == []


def test_risk_analysis_does_not_alert_for_low_risk_detection() -> None:
    detection = make_detection(ObstacleClass.AVE)

    alerts = RiskAnalysisService().analyze([detection])

    assert alerts == []


def test_risk_analysis_alerts_for_high_risk_detection() -> None:
    detection = make_detection(ObstacleClass.FIO_ELETRICO)

    alerts = RiskAnalysisService().analyze([detection])

    assert len(alerts) == 1
    assert alerts[0].detection == detection
    assert alerts[0].risk_level is RiskLevel.ALTO


def test_risk_analysis_raises_medium_risk_for_large_box() -> None:
    detection = make_detection(ObstacleClass.AVE, area=2_500.0)

    alerts = RiskAnalysisService().analyze([detection])

    assert len(alerts) == 1
    assert alerts[0].risk_level is RiskLevel.MEDIO


def test_alert_service_formats_empty_alert_list() -> None:
    assert AlertService().format_alerts([]) == []


def test_alert_service_formats_alert_for_notification() -> None:
    detection = make_detection(ObstacleClass.POSTE)
    alert = Alert(
        detection=detection,
        risk_level=RiskLevel.ALTO,
        message="Poste próximo",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    messages = AlertService().format_alerts([alert])

    assert len(messages) == 1
    assert messages[0] == (
        "[ALTO] Poste próximo (confiança: 90%, "
        "timestamp: 2026-01-01T00:00:00+00:00)"
    )
