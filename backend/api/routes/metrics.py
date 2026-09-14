from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api.dependencies import get_detection_repository, get_metrics_service
from backend.core.config import settings
from backend.repositories.detection_repository import DetectionRepository
from backend.services.metrics_service import MetricsService


router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/report")
def metrics_report(
    annotation_path: Annotated[str | None, Query()] = None,
    repository: DetectionRepository = Depends(get_detection_repository),
    metrics_service: MetricsService = Depends(get_metrics_service),
) -> dict[str, float | int]:
    path = annotation_path or settings.ground_truth_path
    try:
        report = metrics_service.calculate(repository.get_detections(), path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Annotation file not found: {path}",
        ) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return report.to_dict()
