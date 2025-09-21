import copy
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple

from torch import nn

from ..config import ExperimentConfig
from ..pool import DataPool

from transformers import AutoModelForSequenceClassification


class BaseStrategy(ABC):
    """Base class for training strategies."""

    def __init__(self, *,
                 strategy: "BaseStrategy" = None,
                 model: nn.Module = None,
                 optimizer_cls=None, optimizer_kwargs=None,
                 criterion_cls=None, criterion_kwargs=None,
                 scheduler_cls=None, scheduler_kwargs=None,
                 device: str = None, epochs: int = None, batch_size: int = None):
        if strategy is not None:
            # Initialize from another strategy
            self.model = strategy.model
            self.optimizer = strategy.optimizer
            self.criterion = strategy.criterion
            self.scheduler = strategy.scheduler

            self.initial_model_state_dict = strategy.initial_model_state_dict

            self.optimizer_cls = strategy.optimizer_cls
            self.optimizer_kwargs = strategy.optimizer_kwargs
            self.criterion_cls = strategy.criterion_cls
            self.criterion_kwargs = strategy.criterion_kwargs
            self.scheduler_cls = strategy.scheduler_cls
            self.scheduler_kwargs = strategy.scheduler_kwargs

            self.device = strategy.device
            self.epochs = strategy.epochs
            self.batch_size = strategy.batch_size
        else:
            # Store the passed-in class/kwargs
            self.model = model
            self.initial_model_state_dict = copy.deepcopy(model.state_dict()) # Store initial weights for reset

            self.optimizer_cls = optimizer_cls
            self.optimizer_kwargs = optimizer_kwargs
            self.criterion_cls = criterion_cls
            self.criterion_kwargs = criterion_kwargs
            self.scheduler_cls = scheduler_cls
            self.scheduler_kwargs = scheduler_kwargs

            self.device = device
            self.epochs = epochs
            self.batch_size = batch_size
            # self.round_history: List[Dict] = None

            self._initialize_components()

    def train(self, pool: DataPool, new_indices: List[int]) -> Dict:
        """
        Train model for one round with automatic timing.

        Args:
            pool: Current Data Pool
            new_indices: New indices to train the model on (not yet in pool)

        Returns:
            model: Updated model
            stats: Training statistics including timing
        """
        start_time = time.time()

        # Call the strategy-specific training logic
        custom_stats = self._train_implementation(pool, new_indices)

        training_time = time.time() - start_time

        # Add any base statistics...
        base_stats = {
            "training_time": training_time,
        }

        # Merge with strategy-specific stats
        final_stats = {**base_stats, **custom_stats}

        return final_stats

    def _train_batch(self, batch):
        inputs, targets = batch
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
        targets = targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(**inputs)

        logits = outputs['logits']
        loss = self.criterion(logits, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_epochs(self, dataloader) -> tuple[float | Any, int | Any]:
        start_time = time.time()
        total_loss = 0.0
        num_batches = 0

        # Training loop
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_batches = 0
            for batch in dataloader:
                epoch_loss += self._train_batch(batch)
                epoch_batches += 1
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0.0
            logging.info(
                f"Epoch {epoch + 1}/{self.epochs} completed in {epoch_time:.2f}s | Avg Loss: {avg_epoch_loss:.4f}")

            total_loss += epoch_loss
            num_batches += epoch_batches

        total_time = time.time() - start_time
        logging.info(f"Training completed in {total_time:.2f}s")

        return total_loss, num_batches

    def get_stats(self, total_loss, num_batches, tot_samples, new_samples):
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Return model and training statistics
        return {
            "avg_loss": avg_loss,
            "epochs": self.epochs,
            "total_samples": len(tot_samples),
            "new_samples": len(new_samples) if new_samples is not None else 0,
        }

    def _initialize_components(self):
        """Initialize components for training via this strategy."""
        self.model.load_state_dict(self.initial_model_state_dict)
        self.model.to(self.device)
        self.optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)
        self.criterion = self.criterion_cls(**self.criterion_kwargs)

        self.scheduler = None
        if self.scheduler_cls is not None:
            self.scheduler = self.scheduler_cls(self.optimizer, **self.scheduler_kwargs)

    def reset(self):
        """Reset model to initial state by recreating it."""
        logging.info("Resetting model to initial state . . .")
        self._initialize_components()

    @abstractmethod
    def _train_implementation(self, pool: DataPool, new_indices: List[int]) -> Dict:
        """
        Strategy-specific training implementation.

        Returns:
            pool: Current Data Pool
            new_indices: New indices to train the model on (not yet in pool)
        """
        pass