"""Validated runtime settings and user-facing presets."""
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parent

PERFORMANCE_PROFILES = {
    "QUALITY": {"image_size": 640, "inference_fps": 12, "half_precision": False},
    "BALANCED": {"image_size": 512, "inference_fps": 15, "half_precision": False},
    "PERFORMANCE": {"image_size": 416, "inference_fps": 24, "half_precision": True},
}
PROXIMITY_PRESETS = {
    "NEAR": {"proximity_scale": 125, "medium_threshold": 40, "high_threshold": 66,
             "critical_threshold": 86, "approach_threshold": 10},
    "NORMAL": {"proximity_scale": 150, "medium_threshold": 32, "high_threshold": 58,
               "critical_threshold": 78, "approach_threshold": 8},
    "FAR": {"proximity_scale": 175, "medium_threshold": 26, "high_threshold": 50,
            "critical_threshold": 70, "approach_threshold": 6},
}


class Settings(BaseModel):
    """Flat fields preserve the existing HTTP API; presets are optional shortcuts."""

    model_config = {"protected_namespaces": (), "extra": "forbid"}

    # Model
    model_path: str = Field("yolo11n.pt", min_length=1)
    confidence: float = Field(0.45, ge=0.05, le=0.99)
    iou: float = Field(0.45, ge=0.05, le=0.95)
    monitored_classes: list[str] = Field(default_factory=lambda: [
        "person", "car", "truck", "bus", "bicycle", "motorcycle",
        "chair", "bottle", "backpack", "potted plant",
    ])

    # Performance: ceilings, not promises of achieved FPS.
    performance_profile: Literal["QUALITY", "BALANCED", "PERFORMANCE", "CUSTOM"] = "BALANCED"
    image_size: int = Field(512, ge=320, le=1280, multiple_of=32)
    inference_fps: int = Field(15, ge=1, le=60)
    tracking_enabled: bool = True
    device: str = "auto"
    half_precision: bool = False

    # Visual proximity / safety
    proximity_preset: Literal["NEAR", "NORMAL", "FAR", "CUSTOM"] = "NORMAL"
    corridor_width: float = Field(0.52, ge=0.2, le=1.0)
    corridor_height: float = Field(0.64, ge=0.2, le=1.0)
    medium_threshold: float = Field(32, ge=0, le=100)
    high_threshold: float = Field(58, ge=0, le=100)
    critical_threshold: float = Field(78, ge=0, le=100)
    approach_threshold: float = Field(8, ge=0, le=100)
    confirmation_frames: int = Field(3, ge=1, le=12)
    proximity_scale: float = Field(150, ge=50, le=300)
    proximity_smoothing: float = Field(0.35, ge=0.05, le=1.0)

    # Interface / alerts
    overlay_enabled: bool = True
    sound_enabled: bool = True
    alert_cooldown: float = Field(2.5, ge=0, le=30)

    @model_validator(mode="before")
    @classmethod
    def expand_presets(cls, values):
        values = dict(values)
        for field, profiles in (("performance_profile", PERFORMANCE_PROFILES),
                                ("proximity_preset", PROXIMITY_PRESETS)):
            default = "BALANCED" if field == "performance_profile" else "NORMAL"
            preset = profiles.get(values.get(field, default))
            if preset:
                values = {**preset, **values}
                if any(values[key] != value for key, value in preset.items()):
                    values[field] = "CUSTOM"
        return values

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"auto", "cpu", "cuda"} and not (
            value.startswith("cuda:") and value[5:].isdigit()
        ):
            raise ValueError("device must be auto, cpu, cuda or cuda:N")
        return value

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model_path cannot be empty")
        if Path(value).suffix.lower() != ".pt":
            raise ValueError("This runtime currently supports YOLO .pt models")
        return value

    @field_validator("monitored_classes")
    @classmethod
    def normalize_classes(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @model_validator(mode="after")
    def validate_thresholds(self):
        if not self.medium_threshold < self.high_threshold < self.critical_threshold:
            raise ValueError("thresholds must satisfy medium < high < critical")
        return self


def updated_settings(current: Settings, changes: dict) -> Settings:
    """Apply a preset first, then explicit advanced overrides, validating atomically."""
    values = current.model_dump()
    for field, profiles in (("performance_profile", PERFORMANCE_PROFILES),
                            ("proximity_preset", PROXIMITY_PRESETS)):
        if changes.get(field) in profiles:
            values.update(profiles[changes[field]])
    values.update(changes)
    return Settings.model_validate(values)


settings = Settings()
