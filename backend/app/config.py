import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Root directory of the repository (parent of backend)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # NASA Earthdata Login Credentials
    EARTHDATA_USERNAME: str = ""
    EARTHDATA_PASSWORD: str = ""

    # API Configuration
    API_PORT: int = 8000
    API_HOST: str = "127.0.0.1"
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ]

    # Storage paths
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    CASES_DIR: Path = BASE_DIR / "data" / "cases"
    CACHE_DIR: Path = BASE_DIR / "data" / "cache"

    model_config = SettingsConfigDict(
        env_file=[
            str(BACKEND_DIR / ".env"),
            str(BASE_DIR / ".env"),
        ],
        extra="ignore"
    )

settings = Settings()

# Ensure directories exist
for path in [settings.DATA_DIR, settings.RAW_DIR, settings.PROCESSED_DIR, settings.CASES_DIR, settings.CACHE_DIR]:
    path.mkdir(parents=True, exist_ok=True)
