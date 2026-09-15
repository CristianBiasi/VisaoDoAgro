from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field

class Detection(BaseModel):
    track_id: int
    object_class: str
    confidence: float
    bbox: list[float]
    proximity: float
    approach_rate: float
    position: str
    risk: str
    corridor_intersection: float

class Telemetry(BaseModel):
    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None
    accuracy: float | None = None
    speed_kmh: float | None = None
    heading: float | None = None
    pitch: float | None = None
    roll: float | None = None
    timestamp: float | None = None
    source: str = "UNAVAILABLE"

class SettingsPatch(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)

class Event(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    message: str
    level: str = "INFO"
    track_id: int | None = None
