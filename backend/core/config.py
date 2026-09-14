from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    yolo_model_path: str = "models/yolo.pt"
    default_video_source: str = "0"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
