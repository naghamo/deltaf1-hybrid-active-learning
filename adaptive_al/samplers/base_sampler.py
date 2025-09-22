import logging
import random
from abc import ABC, abstractmethod
from typing import List, Set, Any
from ..pool import DataPool

import torch.nn as nn


class BaseSampler(ABC):
    """Base class for active learning samplers."""
    def __init__(self, model: nn.Module, batch_size: int, device: str, seed: int) -> None:
        self.seed = seed

        if seed is not None:
            random.seed(seed)
        else:
            logging.warning("No seed was provided for RandomSampler.")
        self.model = model.to(device)
        self.batch_size = batch_size
        self.device = device

    @abstractmethod
    def select(self,
                   pool: DataPool,
                   num_samples: int) -> List[int]:
        """Select num_samples samples from unlabeled pool."""
        pass

    def get_random_unlabeled(self, pool: DataPool, num_of_indices: int) -> List[int]:
        """Randomly select num_of_indices samples from unlabeled pool."""
        unlabeled_indices = pool.get_unlabeled_indices()

        if len(unlabeled_indices) == 0:
            return []

        # Don't sample more than available
        actual_batch_size = min(num_of_indices, len(unlabeled_indices))

        # Random sampling without replacement
        selected_indices = random.sample(unlabeled_indices, actual_batch_size)

        return selected_indices