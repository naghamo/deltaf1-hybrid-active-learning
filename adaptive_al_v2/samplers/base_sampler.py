from abc import ABC, abstractmethod
from typing import List, Set, Any
from ..pool import DataPool

import torch.nn as nn


class BaseSampler(ABC):
    """Base class for active learning samplers."""
    def __init__(self, model: nn.Module, batch_size: int, device: str):
        self.model = model.to(device)
        self.batch_size = batch_size
        self.device = device

    @abstractmethod
    def select(self,
                   pool: DataPool,
                   num_samples: int) -> List[int]:
        """Select num_samples samples from unlabeled pool."""
        pass
