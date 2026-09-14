import logging
import time
from dataclasses import asdict, dataclass

from backend.infrastructure.video_source import VideoSource
from backend.infrastructure.notifier import Notifier
from backend.repositories.detection_repository import DetectionRepository
from backend.services.alert_service import AlertService
from backend.services.detection_service import DetectionService
from backend.services.risk_analysis_service import RiskAnalysisService


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamProcessingSummary:
    total_frames: int
    average_fps: float
    total_alerts: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class StreamProcessingService:
    def __init__(
        self,
        detection_service: DetectionService,
        risk_analysis_service: RiskAnalysisService,
        alert_service: AlertService,
        notifier: Notifier,
        repository: DetectionRepository | None = None,
    ) -> None:
        self.detection_service = detection_service
        self.risk_analysis_service = risk_analysis_service
        self.alert_service = alert_service
        self.notifier = notifier
        self.repository = repository or DetectionRepository()

    def process(self, video_source: VideoSource) -> StreamProcessingSummary:
        total_frames = 0
        total_alerts = 0
        total_processing_seconds = 0.0

        for frame in video_source.frames():
            frame_started_at = time.perf_counter()
            detections = self.detection_service.detect(frame)
            alerts = self.risk_analysis_service.analyze(detections)
            self.repository.save_results(detections, alerts)
            for alert in alerts:
                self.notifier.publish(alert)
            self.alert_service.format_alerts(alerts)
            frame_processing_seconds = time.perf_counter() - frame_started_at

            total_frames += 1
            total_alerts += len(alerts)
            total_processing_seconds += frame_processing_seconds
            logger.info(
                "Processed frame=%d processing_time=%.4fs",
                total_frames,
                frame_processing_seconds,
            )

        average_fps = (
            total_frames / total_processing_seconds
            if total_processing_seconds > 0
            else 0.0
        )
        logger.info(
            "Finished video source frames=%d average_fps=%.2f total_alerts=%d",
            total_frames,
            average_fps,
            total_alerts,
        )
        return StreamProcessingSummary(
            total_frames=total_frames,
            average_fps=average_fps,
            total_alerts=total_alerts,
        )
