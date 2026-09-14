from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class ObstacleClass(str, Enum):
    ARVORE = "ARVORE"
    POSTE = "POSTE"
    FIO_ELETRICO = "FIO_ELETRICO"
    AVE = "AVE"


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @model_validator(mode="after")
    def validate_coordinates(self) -> "BoundingBox":
        if self.x_min > self.x_max:
            raise ValueError("x_min must be less than or equal to x_max")
        if self.y_min > self.y_max:
            raise ValueError("y_min must be less than or equal to y_max")
        return self


class Detection(BaseModel):
    obstacle_class: ObstacleClass
    confidence: float = Field(ge=0, le=1)
    bounding_box: BoundingBox
    frame_id: str = Field(min_length=1)


class RiskLevel(str, Enum):
    BAIXO = "BAIXO"
    MEDIO = "MEDIO"
    ALTO = "ALTO"


class Alert(BaseModel):
    detection: Detection
    risk_level: RiskLevel
    message: str = Field(min_length=1)
    timestamp: datetime
