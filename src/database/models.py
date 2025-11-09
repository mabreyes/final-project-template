"""Database models for storing MNIST training results."""

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.sql import func

from src.database.base import Base


class TrainingRun(Base):
    """Model for storing training run metadata."""

    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    epochs = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    final_loss = Column(Float, nullable=True)
    final_accuracy = Column(Float, nullable=True)
    training_time_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    notes = Column(Text, nullable=True)


class EpochResult(Base):
    """Model for storing per-epoch training results."""

    __tablename__ = "epoch_results"

    id = Column(Integer, primary_key=True, index=True)
    training_run_id = Column(Integer, nullable=False, index=True)
    epoch = Column(Integer, nullable=False)
    train_loss = Column(Float, nullable=False)
    train_accuracy = Column(Float, nullable=False)
    val_loss = Column(Float, nullable=True)
    val_accuracy = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
