"""CI training script that runs training without database persistence."""

import sys

from src.config import get_training_config
from src.training.trainer import MNISTTrainer

MIN_ACCURACY_THRESHOLD = 90.0


def main():
    """Run training for CI/CD without database."""
    print("=" * 60)
    print("Running Training for CI/CD (No Database)")
    print("=" * 60)

    # Get training configuration
    config = get_training_config()
    print("\nConfiguration:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Device: {config.device}")

    # Create trainer
    trainer = MNISTTrainer(config)

    # Train the model
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    training_results = trainer.train()

    # Display results
    print("\n" + "=" * 60)
    print("Training Results Summary")
    print("=" * 60)
    print(f"Final Validation Loss: {training_results['final_loss']:.4f}")
    print(f"Final Validation Accuracy: {training_results['final_accuracy']:.2f}%")
    print(f"Training Time: {training_results['training_time_seconds']:.2f} seconds")

    print("\n" + "-" * 60)
    print("Per-Epoch Results:")
    print("-" * 60)
    for epoch_result in training_results["epoch_results"]:
        print(
            f"Epoch {epoch_result['epoch']}: "
            f"Train Loss={epoch_result['train_loss']:.4f}, "
            f"Train Acc={epoch_result['train_accuracy']:.2f}%, "
            f"Val Loss={epoch_result['val_loss']:.4f}, "
            f"Val Acc={epoch_result['val_accuracy']:.2f}%"
        )

    print("\n" + "=" * 60)
    print("Training Completed Successfully!")
    print("=" * 60)

    # CI success criteria (optional - can be adjusted)
    if training_results["final_accuracy"] < MIN_ACCURACY_THRESHOLD:
        print(f"\n⚠️  Warning: Final accuracy is below {MIN_ACCURACY_THRESHOLD}%")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
