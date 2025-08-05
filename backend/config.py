import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # Database Configuration
    DATABASE_URL: str = "sqlite:///./botsonic.db"
    
    # JWT Configuration
    JWT_SECRET_KEY: str = "your-secret-key-here-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Groq API Configuration
    GROQ_API_KEY: str = "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"
    
    # ChromaDB Configuration
    CHROMA_DB_PATH: str = "./chroma_db"
    
    # Application Configuration
    APP_NAME: str = "Botsonic Clone"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_EXTENSIONS: list = [".pdf", ".txt", ".xlsx", ".xls"]
    
    # CORS Configuration
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://your-frontend-domain.com"
    ]
    
    # Email Configuration (for future use)
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: Optional[int] = 587
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_USE_TLS: bool = True
    
    # Rate Limiting Configuration
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Selenium Configuration
    SELENIUM_HEADLESS: bool = True
    SELENIUM_TIMEOUT: int = 30
    
    # HuggingFace Model Configuration
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables


# Create settings instance
settings = Settings()


# Environment-specific configurations
def get_database_url() -> str:
    """Get database URL based on environment"""
    if os.getenv("DATABASE_URL"):
        return os.getenv("DATABASE_URL")
    return settings.DATABASE_URL


def get_groq_api_key() -> str:
    """Get Groq API key from environment or settings"""
    return os.getenv("GROQ_API_KEY", settings.GROQ_API_KEY)


def get_jwt_secret_key() -> str:
    """Get JWT secret key from environment or settings"""
    return os.getenv("JWT_SECRET_KEY", settings.JWT_SECRET_KEY)
