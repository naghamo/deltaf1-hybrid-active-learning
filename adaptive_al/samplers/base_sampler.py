from abc import ABC, abstractmethod
from typing import List, Set

class BaseSampler(ABC):
    """
    Abstract base class for active learning sampling methods.
    """
    @abstractmethod
    def select(self, unlabeled_indices: Set[int], model, k: int) -> List[int]:
        """
        Select k samples from the unlabeled pool.

        Args:
            unlabeled_indices: set of indices not yet labeled
            model: nn model
            k: number of samples to select

        Returns:
            List of selected indices
        """
        pass