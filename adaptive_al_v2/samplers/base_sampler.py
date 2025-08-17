from abc import ABC, abstractmethod
from typing import List, Set, Any
from ..pool import DataPool


class BaseSampler(ABC):
    """Base class for active learning samplers."""

    @abstractmethod
    def select(self,
                   pool: DataPool,
                   model: Any,
                   batch_size: int) -> List[int]:
        """Select batch_size samples from unlabeled pool."""
        pass