import os
import tempfile
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.api.dependencies import (
    get_alert_service,
    get_detection_repository,
    get_detection_service,
    get_risk_analysis_service,
)
from backend.domain.entities import Alert, Detection
from backend.infrastructure.yolo_model import YoloModelError
from backend.repositories.detection_repository import DetectionRepository
from backend.services.alert_service import AlertService
from backend.services.detection_service import DetectionService
from backend.services.risk_analysis_service import RiskAnalysisService


router = APIRouter(prefix="/detect", tags=["detection"])


@router.post("/image")
async def detect_image(
    file: Annotated[UploadFile, File(...)],
    detection_service: DetectionService = Depends(get_detection_service),
    risk_analysis_service: RiskAnalysisService = Depends(get_risk_analysis_service),
    alert_service: AlertService = Depends(get_alert_service),
    repository: DetectionRepository = Depends(get_detection_repository),
) -> dict[str, object]:
    image = await _read_image(file)
    detections, alerts, formatted_alerts = _process_frame(
        image,
        detection_service,
        risk_analysis_service,
        alert_service,
        repository,
    )

    return _build_response(detections, alerts, formatted_alerts)


@router.post("/video")
async def detect_video(
    file: Annotated[UploadFile, File(...)],
    detection_service: DetectionService = Depends(get_detection_service),
    risk_analysis_service: RiskAnalysisService = Depends(get_risk_analysis_service),
    alert_service: AlertService = Depends(get_alert_service),
    repository: DetectionRepository = Depends(get_detection_repository),
) -> dict[str, object]:
    video_path = await _save_upload(file)
    detections: list[Detection] = []
    alerts: list[Alert] = []
    frames_processed = 0

    try:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise HTTPException(status_code=400, detail="Could not open uploaded video")

        try:
            while True:
                has_frame, frame = capture.read()
                if not has_frame:
                    break
                frame_detections, frame_alerts, _ = _process_frame(
                    frame,
                    detection_service,
                    risk_analysis_service,
                    alert_service,
                    repository,
                )
                detections.extend(frame_detections)
                alerts.extend(frame_alerts)
                frames_processed += 1
        finally:
            capture.release()
    finally:
        os.unlink(video_path)

    if frames_processed == 0:
        raise HTTPException(status_code=400, detail="Uploaded video has no readable frames")

    formatted_alerts = alert_service.format_alerts(alerts)
    response = _build_response(detections, alerts, formatted_alerts)
    response["frames_processed"] = frames_processed
    return response


async def _read_image(file: UploadFile) -> np.ndarray:
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
    return image


async def _save_upload(file: UploadFile) -> str:
    contents = await file.read()
    suffix = os.path.splitext(file.filename or ".video")[1] or ".video"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
        temporary_file.write(contents)
        return temporary_file.name


def _process_frame(
    frame: np.ndarray,
    detection_service: DetectionService,
    risk_analysis_service: RiskAnalysisService,
    alert_service: AlertService,
    repository: DetectionRepository,
) -> tuple[list[Detection], list[Alert], list[str]]:
    try:
        detections = detection_service.detect(frame)
        alerts = risk_analysis_service.analyze(detections)
        repository.save_results(detections, alerts)
        formatted_alerts = alert_service.format_alerts(alerts)
        return detections, alerts, formatted_alerts
    except YoloModelError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _build_response(
    detections: list[Detection],
    alerts: list[Alert],
    formatted_alerts: list[str],
) -> dict[str, object]:
    return {
        "detections": detections,
        "alerts": alerts,
        "formatted_alerts": formatted_alerts,
        "detection_count": len(detections),
        "alert_count": len(alerts),
    }
