"""Configuration settings for QueueMind FastAPI REST API."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    API_HOST: str = Field(
        default="0.0.0.0",
        description="Host interface for the FastAPI uvicorn server.",
    )
    API_PORT: int = Field(
        default=8000,
        description="Port for the FastAPI uvicorn server.",
    )
    ENVIRONMENT: str = Field(
        default="development",
        description="Runtime environment mode (development, staging, production).",
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Application logging verbosity.",
    )
    ALLOWED_ORIGINS: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="CORS allowed origins for browser clients.",
    )
    PATIENT_FLOW_MODEL_PATH: str | None = Field(
        default=None,
        description="Path to serialized PatientFlowPredictor .joblib artifact.",
    )
    CONGESTION_MODEL_PATH: str | None = Field(
        default=None,
        description="Path to serialized CongestionPredictor .joblib artifact.",
    )

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v: Any) -> list[str]:
        """Support comma-separated string or list for ALLOWED_ORIGINS."""
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                import json

                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return [str(x).strip() for x in parsed]
                except Exception:
                    pass
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        if isinstance(v, list):
            return [str(origin).strip() for origin in v]
        return [str(v)]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton instance of application Settings."""
    return Settings()
