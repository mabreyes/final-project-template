"""MNIST model trainer following SRP."""

import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import TrainingConfig
from src.models.mnist_model import MNISTNet


class MNISTTrainer:
    """Handles MNIST model training."""

    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration."""
        self.config = config
        self.device = torch.device(config.device)
        self.model = MNISTNet().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.training_history: list[dict] = []

    def _get_data_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Get training and validation data loaders."""
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )

        val_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )

        return train_loader, val_loader

    def _train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_inputs, batch_labels in train_loader:
            inputs = batch_inputs.to(self.device)
            labels = batch_labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        return epoch_loss, epoch_accuracy

    def _validate(self, val_loader: DataLoader) -> tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_inputs, batch_labels in val_loader:
                inputs = batch_inputs.to(self.device)
                labels = batch_labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(val_loader)
        epoch_accuracy = 100 * correct / total
        return epoch_loss, epoch_accuracy

    def train(self) -> dict:
        """Train the model and return training results."""
        train_loader, val_loader = self._get_data_loaders()
        start_time = time.time()

        print(f"Starting training for {self.config.epochs} epochs...")

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._validate(val_loader)

            epoch_result = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
            self.training_history.append(epoch_result)

            print(
                f"Epoch {epoch}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

        training_time = time.time() - start_time

        return {
            "final_loss": self.training_history[-1]["val_loss"],
            "final_accuracy": self.training_history[-1]["val_accuracy"],
            "training_time_seconds": training_time,
            "epoch_results": self.training_history,
        }

    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
