import random
from typing import List, Any

from .base_sampler import BaseSampler
from ..pool import DataPool
import logging

class RandomSampler(BaseSampler):
    """Random sampling strategy - selects samples randomly from unlabeled pool."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def select(self, pool: DataPool, acquisition_batch_size: int) -> List[int]:
        return self.get_random_unlabeled(pool, acquisition_batch_size)
