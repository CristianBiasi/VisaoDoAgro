from functools import lru_cache

from fastapi import Depends

from backend.infrastructure.yolo_model import YoloModel
from backend.services.alert_service import AlertService
from backend.services.detection_service import DetectionService
from backend.services.risk_analysis_service import RiskAnalysisService


@lru_cache(maxsize=1)
def get_yolo_model() -> YoloModel:
    return YoloModel()


def get_detection_service(
    model: YoloModel = Depends(get_yolo_model),
) -> DetectionService:
    return DetectionService(model)


def get_risk_analysis_service() -> RiskAnalysisService:
    return RiskAnalysisService()


def get_alert_service() -> AlertService:
    return AlertService()
