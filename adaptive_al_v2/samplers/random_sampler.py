import random
from typing import List, Any

from .base_sampler import BaseSampler
from ..pool import DataPool
import logging

class RandomSampler(BaseSampler):
    """Random sampling strategy - selects samples randomly from unlabeled pool."""

    def __init__(self, seed: int = None, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed

        if seed is not None:
            random.seed(seed)
        else:
            logging.warning("No seed was provided for RandomSampler.")

    def select(self, pool: DataPool, acquisition_batch_size: int) -> List[int]:
        """Randomly select batch_size samples from unlabeled pool."""
        unlabeled_indices = pool.get_unlabeled_indices()

        if len(unlabeled_indices) == 0:
            return []

        # Don't sample more than available
        actual_batch_size = min(acquisition_batch_size, len(unlabeled_indices))

        # Random sampling without replacement
        selected_indices = random.sample(unlabeled_indices, actual_batch_size)

        return selected_indices