from pathlib import Path
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent

class Settings(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_path: str = "yolo11n.pt"
    confidence: float = Field(0.45, ge=0.05, le=0.99)
    iou: float = Field(0.45, ge=0.05, le=0.95)
    image_size: int = Field(416, ge=320, le=1280)
    inference_fps: int = Field(4, ge=1, le=30)
    tracking_enabled: bool = True
    overlay_enabled: bool = True
    monitored_classes: list[str] = ["person", "car", "truck", "bus", "bicycle", "motorcycle", "chair", "bottle", "backpack", "potted plant"]
    corridor_width: float = Field(0.52, ge=0.2, le=1.0)
    corridor_height: float = Field(0.64, ge=0.2, le=1.0)
    medium_threshold: float = Field(32, ge=0, le=100)
    high_threshold: float = Field(58, ge=0, le=100)
    critical_threshold: float = Field(78, ge=0, le=100)
    approach_threshold: float = Field(8, ge=0, le=100)
    confirmation_frames: int = Field(3, ge=1, le=12)
    proximity_scale: float = Field(150, ge=50, le=300)
    proximity_smoothing: float = Field(0.35, ge=0.05, le=1.0)
    sound_enabled: bool = True
    alert_cooldown: float = Field(2.5, ge=0, le=30)

settings = Settings()
