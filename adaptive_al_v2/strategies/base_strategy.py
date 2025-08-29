import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple

from torch import nn

from ..config import ExperimentConfig
from ..pool import DataPool


class BaseStrategy(ABC):
    """Base class for training strategies."""

    def __init__(self,
                 model: nn.Module, # CHANGED: from model_cls, model_kwargs
                 optimizer_cls, optimizer_kwargs,
                 criterion_cls, criterion_kwargs,
                 scheduler_cls, scheduler_kwargs,
                 device, epochs, batch_size):

        # Store the passed-in model instance directly
        self.model = model
        self.initial_model_state_dict = model.state_dict() # Store initial weights for reset

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_cls = criterion_cls
        self.criterion_kwargs = criterion_kwargs
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs

        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size

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

    def _initialize_components(self):
        """Initialize components for training via this straegy."""
        self.model.to(self.device)
        self.optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)
        self.criterion = self.criterion_cls(**self.criterion_kwargs)

        self.scheduler = None
        if self.scheduler_cls is not None:
            self.scheduler = self.scheduler_cls(self.optimizer, **self.scheduler_kwargs)

        self.round_history = []

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