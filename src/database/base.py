"""Database base setup and session management."""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from src.config import DatabaseConfig, get_database_config

Base = declarative_base()


class Database:
    """Database connection manager following SRP."""

    def __init__(self, config: DatabaseConfig):
        """Initialize database with configuration."""
        self.config = config
        self.engine = create_engine(
            config.get_database_url(), pool_pre_ping=True, echo=False
        )
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def get_session(self) -> Generator[Session, None, None]:
        """Get database session generator."""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()

    def create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(bind=self.engine)


def get_database() -> Database:
    """Get database instance with dependency injection."""
    config = get_database_config()
    return Database(config)
