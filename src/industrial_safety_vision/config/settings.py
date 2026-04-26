"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Runtime settings for local demos and API service."""

    model_path: str = "models/best.pt"
    inference_confidence: float = 0.35
    inference_iou: float = 0.45
    device: str = "auto"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = AppSettings()
