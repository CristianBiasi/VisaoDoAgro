from functools import lru_cache

from fastapi import Depends

from backend.infrastructure.notifier import Notifier
from backend.infrastructure.yolo_model import YoloModel
from backend.repositories.detection_repository import DetectionRepository
from backend.services.alert_service import AlertService
from backend.services.detection_service import DetectionService
from backend.services.risk_analysis_service import RiskAnalysisService
from backend.services.stream_processing_service import StreamProcessingService
from backend.services.metrics_service import MetricsService


@lru_cache(maxsize=1)
def get_yolo_model() -> YoloModel:
    return YoloModel()


@lru_cache(maxsize=1)
def get_notifier() -> Notifier:
    return Notifier()


@lru_cache(maxsize=1)
def get_detection_repository() -> DetectionRepository:
    from backend.core.config import settings

    return DetectionRepository(settings.database_path)


def get_detection_service(
    model: YoloModel = Depends(get_yolo_model),
) -> DetectionService:
    return DetectionService(model)


def get_risk_analysis_service() -> RiskAnalysisService:
    return RiskAnalysisService()


def get_alert_service() -> AlertService:
    return AlertService()


def get_metrics_service() -> MetricsService:
    return MetricsService()


def get_stream_processing_service(
    detection_service: DetectionService = Depends(get_detection_service),
    risk_analysis_service: RiskAnalysisService = Depends(get_risk_analysis_service),
    alert_service: AlertService = Depends(get_alert_service),
    notifier: Notifier = Depends(get_notifier),
    repository: DetectionRepository = Depends(get_detection_repository),
) -> StreamProcessingService:
    return StreamProcessingService(
        detection_service=detection_service,
        risk_analysis_service=risk_analysis_service,
        alert_service=alert_service,
        notifier=notifier,
        repository=repository,
    )
