import torch
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

    def _train_implementation(self, pool: DataPool, new_indices: List[int]) -> Dict:
        pass