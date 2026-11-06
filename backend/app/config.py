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

class Settings(BaseModel):
    PROJECT_NAME: str = "Fashion Item Classification with CNN"
    PROJECT_ID: str = "DL-002"
    DOMAIN: str = "Computer Vision / Retail"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # Local frontend origins for CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]

settings = Settings()
