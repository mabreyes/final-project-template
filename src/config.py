"""Configuration management using Pydantic settings."""

from typing import Optional

from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""

    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "mnist_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: Optional[str] = None

    def get_database_url(self) -> str:
        """Get database URL, either from environment or construct from components."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    class Config:
        env_file = ".env"
        env_prefix = "POSTGRES_"
        case_sensitive = False


class TrainingConfig(BaseSettings):
    """Training configuration settings."""

    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 5
    device: str = "cpu"  # 'cpu' or 'cuda'

    class Config:
        env_file = ".env"
        env_prefix = "TRAINING_"
        case_sensitive = False


def get_database_config() -> DatabaseConfig:
    """Get database configuration instance."""
    return DatabaseConfig()


def get_training_config() -> TrainingConfig:
    """Get training configuration instance."""
    return TrainingConfig()
