import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple

from .base_strategy import BaseStrategy
from ..pool import DataPool


class NewOnlyStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _train_implementation(self, pool: DataPool, new_indices: List[int]) -> Dict:
        """
        Fine-tunes the model for epochs using only the new data given by the oracle.
        """
        if new_indices:
            pool.add_labeled_samples(new_indices)

        labeled_subset = pool.get_labeled_subset()
        new_labeled_subset = pool.get_subset(new_indices)

        dataloader = DataLoader(new_labeled_subset, batch_size=self.batch_size, shuffle=True)

        total_loss, num_batches = self.train_epochs(dataloader)

        return self.get_stats(total_loss, num_batches, labeled_subset, new_indices)

