from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "HarvestLenz"
    APP_ENV: str = "development"
    SECRET_KEY: str = "change-me-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    DATABASE_URL: str = "sqlite+aiosqlite:///./test_db.sqlite"

    STORAGE_TYPE: str = "local"
    STORAGE_BASE_PATH: str = "storage"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_BUCKET_NAME: Optional[str] = None
    AWS_REGION: str = "ap-south-1"

    GRADES_MODEL_DIR: str = "app/models/weights"
    SUPPORTED_FRUITS: list = ["mango", "pineapple", "grapes", "pomegranate", "orange", "guava", "kiwi", "watermelon", "banana", "cocoa", "coffee", "strawberry", "plum", "peach", "pear"]
    LOG_LEVEL: str = "INFO"
    DEMO_MODE: bool = False

    class Config:
        env_file = ".env"

settings = Settings()
