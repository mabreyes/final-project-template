"""Service layer for coordinating training and database operations."""

from sqlalchemy.orm import Session

from src.config import get_training_config
from src.database.models import EpochResult, TrainingRun
from src.database.repository import EpochResultRepository, TrainingRunRepository
from src.training.trainer import MNISTTrainer


class TrainingService:
    """Service for coordinating training and persistence."""

    def __init__(
        self,
        trainer: MNISTTrainer,
        training_run_repo: TrainingRunRepository,
        epoch_result_repo: EpochResultRepository,
    ):
        """Initialize service with dependencies injected."""
        self.trainer = trainer
        self.training_run_repo = training_run_repo
        self.epoch_result_repo = epoch_result_repo

    def train_and_save(
        self, model_name: str = "MNISTNet", notes: str | None = None
    ) -> TrainingRun:
        """Train model and save results to database."""
        # Train the model
        training_results = self.trainer.train()

        # Create training run record
        training_run = TrainingRun(
            model_name=model_name,
            epochs=self.trainer.config.epochs,
            batch_size=self.trainer.config.batch_size,
            learning_rate=self.trainer.config.learning_rate,
            final_loss=training_results["final_loss"],
            final_accuracy=training_results["final_accuracy"],
            training_time_seconds=training_results["training_time_seconds"],
            notes=notes,
        )

        # Save training run
        saved_run = self.training_run_repo.create(training_run)

        # Save epoch results
        for epoch_result in training_results["epoch_results"]:
            epoch_record = EpochResult(
                training_run_id=saved_run.id,
                epoch=epoch_result["epoch"],
                train_loss=epoch_result["train_loss"],
                train_accuracy=epoch_result["train_accuracy"],
                val_loss=epoch_result["val_loss"],
                val_accuracy=epoch_result["val_accuracy"],
            )
            self.epoch_result_repo.create(epoch_record)

        print(f"\nTraining run saved to database with ID: {saved_run.id}")
        return saved_run


def create_training_service(session: Session) -> TrainingService:
    """Factory function to create TrainingService with dependencies."""
    config = get_training_config()
    trainer = MNISTTrainer(config)
    training_run_repo = TrainingRunRepository(session)
    epoch_result_repo = EpochResultRepository(session)
    return TrainingService(trainer, training_run_repo, epoch_result_repo)
