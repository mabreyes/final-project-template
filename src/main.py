"""Main entry point for the application."""

from src.database.base import get_database
from src.services.training_service import create_training_service


def main():
    """Main function to run training and save to database."""
    # Get database instance
    db = get_database()

    # Get database session
    session_gen = db.get_session()
    session = next(session_gen)

    try:
        # Create training service with dependency injection
        training_service = create_training_service(session)

        # Train and save results
        training_run = training_service.train_and_save(
            model_name="MNISTNet", notes="Initial training run"
        )

        print("\nTraining completed successfully!")
        print(f"Training Run ID: {training_run.id}")
        print(f"Final Accuracy: {training_run.final_accuracy:.2f}%")
        print(f"Final Loss: {training_run.final_loss:.4f}")

    except Exception as e:
        session.rollback()
        print(f"Error during training: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
