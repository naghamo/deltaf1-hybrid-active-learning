"""
Random sampling baseline for active learning.

This module implements a simple random sampling strategy that serves as
a baseline for comparing more sophisticated acquisition functions. Samples
are selected uniformly at random from the unlabeled pool without using
any model predictions or heuristics.
"""

import random
from typing import List, Any

from .base_sampler import BaseSampler
from ..pool import DataPool
import logging

class RandomSampler(BaseSampler):
    """
    Random sampling strategy that selects samples uniformly at random.
    """

    def __init__(self, **kwargs):
        """
        Initialize the random sampler.

        Args:
            **kwargs: Arguments passed to BaseSampler (model, batch_size, device, seed).
        """
        super().__init__(**kwargs)

    def select(self, pool: DataPool, acquisition_batch_size: int) -> List[int]:
        """
        Randomly select samples from the unlabeled pool.

        Selects samples uniformly at random without using any model predictions
        or heuristics. Returns fewer samples if the requested number exceeds
        available unlabeled data.

        Args:
            pool (DataPool): Data pool containing labeled and unlabeled samples.
            acquisition_batch_size (int): Number of samples to select.

        Returns:
            List[int]: Randomly selected indices from the unlabeled pool.
        """
        return self.get_random_unlabeled(pool, acquisition_batch_size)
