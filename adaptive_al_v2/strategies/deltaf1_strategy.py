import torch
from sympy.physics.paulialgebra import epsilon
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple

from .base_strategy import BaseStrategy
from ..pool import DataPool

from .fine_tuning_strategy import FineTuneStrategy
from .retrain_strategy import RetrainStrategy


class DeltaF1Strategy(BaseStrategy):
    def __init__(self, epsilon: float, k: int, **kwargs):
        super().__init__(**kwargs)

        # Or anything else specific to the deltaf1 method . . . then pass those in kwargs of cfg
        self.epsilon = epsilon
        self.k = k

    def calc_f1(self):
        raise NotImplementedError

    def _train_implementation(self, pool: DataPool, new_indices: List[int]) -> Dict:
        if new_indices:
            pool.add_labeled_samples(new_indices)
        labeled_subset = pool.get_labeled_subset()
        dataloader = DataLoader(labeled_subset, batch_size=self.batch_size, shuffle=True)

        if self.calc_f1() < self.epsilon:
            self._reset_model()

        total_loss, num_batches = self._train_epochs(dataloader)

        return self._get_stats(total_loss, num_batches, len(labeled_subset), len(new_indices))