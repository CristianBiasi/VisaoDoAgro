from datetime import datetime, timezone

from backend.domain.entities import Alert, Detection, ObstacleClass, RiskLevel


class RiskAnalysisService:
    """Applies class and bounding-box rules to identify risky detections."""

    _CLASS_SCORES = {
        ObstacleClass.AVE: 1,
        ObstacleClass.ARVORE: 2,
        ObstacleClass.POSTE: 3,
        ObstacleClass.FIO_ELETRICO: 3,
    }
    _RISK_LEVELS = {
        1: RiskLevel.BAIXO,
        2: RiskLevel.MEDIO,
        3: RiskLevel.ALTO,
    }

    def __init__(
        self,
        minimum_risk: RiskLevel = RiskLevel.MEDIO,
        medium_area_threshold: float = 2_500.0,
        high_area_threshold: float = 10_000.0,
    ) -> None:
        if medium_area_threshold < 0 or high_area_threshold < medium_area_threshold:
            raise ValueError("Bounding-box area thresholds are invalid")

        self.minimum_risk = minimum_risk
        self.medium_area_threshold = medium_area_threshold
        self.high_area_threshold = high_area_threshold

    def analyze(self, detections: list[Detection]) -> list[Alert]:
        alerts: list[Alert] = []
        for detection in detections:
            risk_level = self._calculate_risk(detection)
            if self._meets_minimum_risk(risk_level):
                alerts.append(
                    Alert(
                        detection=detection,
                        risk_level=risk_level,
                        message=self._build_message(detection, risk_level),
                        timestamp=datetime.now(timezone.utc),
                    )
                )
        return alerts

    def _calculate_risk(self, detection: Detection) -> RiskLevel:
        score = self._CLASS_SCORES[detection.obstacle_class]
        area = self._bounding_box_area(detection)

        if area >= self.high_area_threshold:
            score = max(score, 3)
        elif area >= self.medium_area_threshold:
            score = max(score, 2)

        return self._RISK_LEVELS[score]

    @staticmethod
    def _bounding_box_area(detection: Detection) -> float:
        box = detection.bounding_box
        return (box.x_max - box.x_min) * (box.y_max - box.y_min)

    def _meets_minimum_risk(self, risk_level: RiskLevel) -> bool:
        risk_order = {
            RiskLevel.BAIXO: 1,
            RiskLevel.MEDIO: 2,
            RiskLevel.ALTO: 3,
        }
        return risk_order[risk_level] >= risk_order[self.minimum_risk]

    @staticmethod
    def _build_message(detection: Detection, risk_level: RiskLevel) -> str:
        return (
            f"Obstáculo {detection.obstacle_class.value} com risco "
            f"{risk_level.value.lower()} no frame {detection.frame_id}."
        )
