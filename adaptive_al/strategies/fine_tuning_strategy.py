import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple

from .base_strategy import BaseStrategy
from ..pool import DataPool


class FineTuneStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _train_implementation(self, pool: DataPool, new_indices: List[int]) -> Dict:
        """
        Fine-tunes the model for epochs using all labeled data so far.
        """
        self.model.train()  # Set the model to training mode
        self.model.to(self.device)

        if new_indices:
            pool.add_labeled_samples(new_indices)

        # Get all labeled data from the pool
        labeled_subset = pool.get_labeled_subset()
        dataloader = DataLoader(labeled_subset, batch_size=self.batch_size, shuffle=True)
        total_loss, num_batches = self.train_epochs(dataloader)
        return self.get_stats(total_loss, num_batches, labeled_subset, new_indices)
