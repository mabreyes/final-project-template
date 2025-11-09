"""Repository pattern for database operations following SRP."""

from typing import Optional

from sqlalchemy.orm import Session

from src.database.models import EpochResult, TrainingRun


class TrainingRunRepository:
    """Repository for TrainingRun operations."""

    def __init__(self, session: Session):
        """Initialize repository with database session."""
        self.session = session

    def create(self, training_run: TrainingRun) -> TrainingRun:
        """Create a new training run."""
        self.session.add(training_run)
        self.session.commit()
        self.session.refresh(training_run)
        return training_run

    def get_by_id(self, run_id: int) -> Optional[TrainingRun]:
        """Get training run by ID."""
        return self.session.query(TrainingRun).filter(TrainingRun.id == run_id).first()

    def get_all(self) -> list[TrainingRun]:
        """Get all training runs."""
        return (
            self.session.query(TrainingRun)
            .order_by(TrainingRun.created_at.desc())
            .all()
        )


class EpochResultRepository:
    """Repository for EpochResult operations."""

    def __init__(self, session: Session):
        """Initialize repository with database session."""
        self.session = session

    def create(self, epoch_result: EpochResult) -> EpochResult:
        """Create a new epoch result."""
        self.session.add(epoch_result)
        self.session.commit()
        self.session.refresh(epoch_result)
        return epoch_result

    def get_by_training_run_id(self, training_run_id: int) -> list[EpochResult]:
        """Get all epoch results for a training run."""
        return (
            self.session.query(EpochResult)
            .filter(EpochResult.training_run_id == training_run_id)
            .order_by(EpochResult.epoch)
            .all()
        )
