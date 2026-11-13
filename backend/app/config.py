import os
from pathlib import Path
from pydantic import BaseModel

# Base directory for the backend
BACKEND_DIR = Path(__file__).resolve().parent.parent

# Standardized folder paths
DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
OOD_ARTIFACTS_DIR = BACKEND_DIR / "ood_artifacts"
OOD_CANDIDATE_MODEL_PATH = OOD_ARTIFACTS_DIR / "robustness_candidate" / "ood_detector_candidate.keras"

# Calibrated OOD Decision Thresholds (Stage 4.1.1 Audited)
DEFAULT_OOD_REJECT_THRESHOLD: float = 0.02
DEFAULT_OOD_ACCEPT_THRESHOLD: float = 0.98


def validate_ood_thresholds(reject_threshold: float, accept_threshold: float) -> None:
    """
    Validates calibrated OOD decision thresholds:
    Must satisfy: 0.0 <= reject_threshold < accept_threshold <= 1.0.
    Raises ValueError if thresholds are inconsistent or invalid.
    """
    if not (0.0 <= reject_threshold < accept_threshold <= 1.0):
        raise ValueError(
            f"Invalid OOD thresholds: reject_threshold ({reject_threshold}) and "
            f"accept_threshold ({accept_threshold}) must satisfy 0.0 <= reject < accept <= 1.0."
        )

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

# Default origins (local development and verified production frontend)
DEFAULT_CORS_ORIGINS: list[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "https://fashion-item-classification.vercel.app",
]


def resolve_cors_origins(raw_env: str | None = None) -> list[str]:
    """
    Resolves allowed CORS origins from an optional comma-separated environment variable (CORS_ORIGINS).
    Trims whitespace and discards empty entries.
    If the variable is absent or empty, preserves default local development origins and production Vercel frontend.
    """
    if raw_env is None:
        raw_env = os.getenv("CORS_ORIGINS")
    if not raw_env:
        return list(DEFAULT_CORS_ORIGINS)

    parsed = [origin.strip() for origin in raw_env.split(",") if origin.strip()]
    for default_origin in DEFAULT_CORS_ORIGINS:
        if default_origin not in parsed:
            parsed.append(default_origin)
    return parsed


class Settings(BaseModel):
    PROJECT_NAME: str = "Fashion Item Classification with CNN"
    PROJECT_ID: str = "DL-002"
    DOMAIN: str = "Computer Vision / Retail"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    # Out-of-Distribution (OOD) Gate Configuration
    OOD_MODEL_PATH: Path = OOD_CANDIDATE_MODEL_PATH
    OOD_REJECT_THRESHOLD: float = DEFAULT_OOD_REJECT_THRESHOLD
    OOD_ACCEPT_THRESHOLD: float = DEFAULT_OOD_ACCEPT_THRESHOLD

    @property
    def CORS_ORIGINS(self) -> list[str]:
        return resolve_cors_origins()


settings = Settings()
