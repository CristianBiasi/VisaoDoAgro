from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    yolo_model_path: str = "models/yolo.pt"
    default_video_source: str = "0"
    log_level: str = "INFO"
    database_path: str = "data/detections.db"
    ground_truth_path: str = "tests/fixtures/ground_truth.json"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
