import os
from pathlib import Path
from pydantic import BaseModel

# Base directory for the backend
BACKEND_DIR = Path(__file__).resolve().parent.parent

# Standardized folder paths
DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"

# Fashion-MNIST 10 Canonical Class Labels
FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Default local development origins
DEFAULT_CORS_ORIGINS: list[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]


def resolve_cors_origins(raw_env: str | None = None) -> list[str]:
    """
    Resolves allowed CORS origins from an optional comma-separated environment variable (CORS_ORIGINS).
    Trims whitespace and discards empty entries.
    If the variable is absent or empty, preserves default local development origins.
    """
    if raw_env is None:
        raw_env = os.getenv("CORS_ORIGINS")
    if not raw_env:
        return list(DEFAULT_CORS_ORIGINS)

    parsed = [origin.strip() for origin in raw_env.split(",") if origin.strip()]
    return parsed if parsed else list(DEFAULT_CORS_ORIGINS)


class Settings(BaseModel):
    PROJECT_NAME: str = "Fashion Item Classification with CNN"
    PROJECT_ID: str = "DL-002"
    DOMAIN: str = "Computer Vision / Retail"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    @property
    def CORS_ORIGINS(self) -> list[str]:
        return resolve_cors_origins()


settings = Settings()
