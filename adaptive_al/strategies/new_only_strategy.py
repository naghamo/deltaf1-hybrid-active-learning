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
        if not new_indices:
            return self.get_stats(0, 0, pool.get_subset([]), [])

        pool.add_labeled_samples(new_indices)

        labeled_subset = pool.get_subset(new_indices)
        dataloader = DataLoader(labeled_subset, batch_size=len(new_indices), shuffle=True)


        total_loss, num_batches = self.train_epochs(dataloader)

        return self.get_stats(total_loss, num_batches, labeled_subset, new_indices)

