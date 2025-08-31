import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple

from .base_strategy import BaseStrategy
from ..pool import DataPool


class RetrainStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _train_implementation(self, pool: DataPool, new_indices: List[int]) -> Dict:
        """
        Resets the model and retrains it for epochs using all labeled data so far.
        """

        if new_indices:
            pool.add_labeled_samples(new_indices)
        labeled_subset = pool.get_labeled_subset()
        dataloader = DataLoader(labeled_subset, batch_size=self.batch_size, shuffle=True)

        self._reset_model()

        total_loss, num_batches = self._train_epochs(dataloader)

        return self._get_stats(total_loss, num_batches, len(labeled_subset), len(new_indices))
