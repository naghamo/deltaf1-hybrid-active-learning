import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple

from .base_strategy import BaseStrategy
from ..pool import DataPool


class NewOnlyStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _train_implementation(self, pool: DataPool, new_indices: List[int]) -> Dict:
        pass
