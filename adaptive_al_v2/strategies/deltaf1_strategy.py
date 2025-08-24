import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple

from .base_strategy import BaseStrategy
from ..pool import DataPool


class DeltaF1Strategy(BaseStrategy):
    def __init__(self, epsilon: float, k: int, **kwargs):
        super().__init__(**kwargs)

        # Or anything else specific to the deltaf1 method . . .
        self.epsilon = epsilon
        self.k = k

    def _train_implementation(self, model: Any, pool, new_indices: List[int]) -> Tuple[Any, Dict]:
        pass