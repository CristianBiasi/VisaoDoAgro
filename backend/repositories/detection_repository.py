import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from backend.domain.entities import Alert, BoundingBox, Detection, ObstacleClass, RiskLevel


class DetectionRepository:
    """SQLite repository for detections and alerts produced by the pipeline."""

    def __init__(self, database_path: str | Path = "data/detections.db") -> None:
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = Lock()
        self._create_tables()

    def save_detection(
        self,
        detection: Detection,
        timestamp: datetime | None = None,
    ) -> int:
        with self._lock, self._connection:
            return self._insert_detection(detection, timestamp)

    def save_alert(self, alert: Alert, timestamp: datetime | None = None) -> int:
        with self._lock, self._connection:
            detection_id = self._insert_detection(alert.detection, timestamp)
            return self._insert_alert(alert, detection_id, timestamp)

    def save_results(
        self,
        detections: list[Detection],
        alerts: list[Alert],
    ) -> None:
        with self._lock, self._connection:
            detection_ids = {
                self._detection_key(detection): self._insert_detection(detection)
                for detection in detections
            }
            for alert in alerts:
                key = self._detection_key(alert.detection)
                detection_id = detection_ids.get(key)
                if detection_id is None:
                    detection_id = self._insert_detection(alert.detection)
                self._insert_alert(alert, detection_id)

    def get_detections(self) -> list[Detection]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT obstacle_class, confidence, x_min, y_min, x_max, y_max, frame_id
                FROM detections
                ORDER BY id
                """
            ).fetchall()
        return [self._row_to_detection(row) for row in rows]

    def count_alerts(self) -> int:
        with self._lock:
            row = self._connection.execute("SELECT COUNT(*) AS count FROM alerts").fetchone()
        return int(row["count"])

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _create_tables(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    obstacle_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    x_min REAL NOT NULL,
                    y_min REAL NOT NULL,
                    x_max REAL NOT NULL,
                    y_max REAL NOT NULL,
                    frame_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detection_id INTEGER NOT NULL,
                    risk_level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    frame_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (detection_id) REFERENCES detections(id)
                );
                """
            )

    def _insert_detection(
        self,
        detection: Detection,
        timestamp: datetime | None = None,
    ) -> int:
        cursor = self._connection.execute(
            """
            INSERT INTO detections (
                obstacle_class, confidence, x_min, y_min, x_max, y_max, frame_id, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                detection.obstacle_class.value,
                detection.confidence,
                detection.bounding_box.x_min,
                detection.bounding_box.y_min,
                detection.bounding_box.x_max,
                detection.bounding_box.y_max,
                detection.frame_id,
                self._timestamp(timestamp),
            ),
        )
        return int(cursor.lastrowid)

    def _insert_alert(
        self,
        alert: Alert,
        detection_id: int,
        timestamp: datetime | None = None,
    ) -> int:
        cursor = self._connection.execute(
            """
            INSERT INTO alerts (
                detection_id, risk_level, message, frame_id, timestamp
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                detection_id,
                alert.risk_level.value,
                alert.message,
                alert.detection.frame_id,
                self._timestamp(timestamp or alert.timestamp),
            ),
        )
        return int(cursor.lastrowid)

    @staticmethod
    def _detection_key(detection: Detection) -> tuple[object, ...]:
        box = detection.bounding_box
        return (
            detection.frame_id,
            detection.obstacle_class,
            box.x_min,
            box.y_min,
            box.x_max,
            box.y_max,
        )

    @staticmethod
    def _timestamp(timestamp: datetime | None) -> str:
        return (timestamp or datetime.now(timezone.utc)).isoformat()

    @staticmethod
    def _row_to_detection(row: sqlite3.Row) -> Detection:
        return Detection(
            obstacle_class=ObstacleClass(row["obstacle_class"]),
            confidence=row["confidence"],
            bounding_box=BoundingBox(
                x_min=row["x_min"],
                y_min=row["y_min"],
                x_max=row["x_max"],
                y_max=row["y_max"],
            ),
            frame_id=row["frame_id"],
        )
