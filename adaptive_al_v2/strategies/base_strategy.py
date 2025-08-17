import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple

from ..config import ExperimentConfig
from ..pool import DataPool


class BaseStrategy(ABC):
    """Base class for training strategies."""

    def __init__(self, optimizer, criterion, scheduler, device, epochs, batch_size):
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size

        self.round_history: List[Dict] = []

    def train(self, model: Any, pool: DataPool, new_indices: List[int]) -> Tuple[Any, Dict]:
        """
        Train model for one round with automatic timing.

        Args:
            model: Current model to train
            pool: Current Data Pool
            new_indices: New indices to train the model on

        Returns:
            model: Updated model
            stats: Training statistics including timing
        """
        start_time = time.time()

        # Call the strategy-specific training logic
        model, custom_stats = self._train_implementation(model, pool, new_indices)

        training_time = time.time() - start_time

        # Add any base statistics...
        base_stats = {
            "training_time": training_time,
        }

        # Merge with strategy-specific stats
        final_stats = {**base_stats, **custom_stats}

        return model, final_stats

    @abstractmethod
    def _train_implementation(self, model: Any, pool: DataPool, new_indices: List[int]) -> Tuple[Any, Dict]:
        """
        Strategy-specific training implementation.

        Returns:
            model: Current model to train
            pool: Current Data Pool
            new_indices: New indices to train the model on
        """
        pass