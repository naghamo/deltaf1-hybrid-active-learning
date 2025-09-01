import torch
from sympy.physics.paulialgebra import epsilon
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple

from .base_strategy import BaseStrategy
from ..pool import DataPool

from ..evaluation import evaluate_model

from .fine_tuning_strategy import FineTuneStrategy
from .retrain_strategy import RetrainStrategy

class DeltaF1Strategy(BaseStrategy):
    def __init__(self, epsilon: float, k: int, **kwargs):
        super().__init__(**kwargs)

        # Or anything else specific to the deltaf1 method . . . then pass those in kwargs of cfg
        self.epsilon = epsilon
        self.k = k

        self.count = 0
        self.switched = False
        self.prev_f1 = None

        self.fine_tune = FineTuneStrategy(**kwargs)
        self.retrain = RetrainStrategy(**kwargs)

    def _calc_f1(self, subset_to_evaluate):
        metrics = evaluate_model(self.model, self.criterion, self.batch_size, subset_to_evaluate, self.device)
        return metrics['f1_score']

    def _train_implementation(self, pool: DataPool, new_indices: List[int]) -> Dict:
        if new_indices:
            pool.add_labeled_samples(new_indices)
        labeled_subset = pool.get_labeled_subset()
        dataloader = DataLoader(labeled_subset, batch_size=self.batch_size, shuffle=True)

        if self.switched:
            total_loss, num_batches = self.fine_tune.train_epochs(dataloader)
        else:
            total_loss, num_batches = self.retrain.train_epochs(dataloader)

            # On what to evaluate???
            cur_f1 = self._calc_f1(labeled_subset)
            delta_f1 = cur_f1 - self.prev_f1 if self.prev_f1 else 0

            if abs(delta_f1) < self.epsilon:
                self.count += 1
            else:
                self.count = 0

            if self.count >= self.k and not self.switched:
                self.switched = True

        return self.get_stats(total_loss, num_batches, labeled_subset, new_indices)