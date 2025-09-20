from .entropy_sampler import EntropySampler
from ..pool import DataPool
import random


class EntropyOnRandomSubsetSampler(EntropySampler):
    """Selects samples with the highest predictive entropy."""

    def __init__(self, random_subset_size, **kwargs):
        super().__init__(**kwargs)
        self.random_subset_size = random_subset_size


    def get_indices_and_subset(self, pool: DataPool):
        unlabeled_indices = pool.get_unlabeled_indices()
        if len(unlabeled_indices) > self.random_subset_size:
            unlabeled_indices = random.sample(unlabeled_indices, self.random_subset_size)
        return unlabeled_indices, pool.get_subset(unlabeled_indices)
