from backend.domain.entities import Alert


class AlertService:
    def format_alerts(self, alerts: list[Alert]) -> list[str]:
        return [self._format_alert(alert) for alert in alerts]

    @staticmethod
    def _format_alert(alert: Alert) -> str:
        detection = alert.detection
        return (
            f"[{alert.risk_level.value}] {alert.message} "
            f"(confiança: {detection.confidence:.0%}, "
            f"timestamp: {alert.timestamp.isoformat()})"
        )
