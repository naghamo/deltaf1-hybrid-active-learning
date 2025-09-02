from typing import List, Dict, Any, Tuple, Optional, Set
from torch.utils.data import Subset
from scipy.stats import entropy

class DataPool:
    """Manages labeled and unlabeled data pools."""

    def __init__(self, train_dataset, val_dataset, test_dataset, initial_labeled_indices: List[int]):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # Convert to sets for efficient operations
        self.labeled_indices: Set[int] = set(initial_labeled_indices)
        self.unlabeled_indices: Set[int] = set(range(len(train_dataset))) - self.labeled_indices

        # History tracking
        self.history: List[Dict] = []

    def add_labeled_samples(self, new_indices: List[int]) -> None:
        """Add new samples to labeled pool and remove from unlabeled."""
        new_indices_set = set(new_indices)

        # Validate indices are currently unlabeled
        invalid_indices = new_indices_set - self.unlabeled_indices
        if invalid_indices:
            raise ValueError(f"Indices {invalid_indices} are not in unlabeled pool")

        # Update pools
        self.labeled_indices.update(new_indices_set)
        self.unlabeled_indices -= new_indices_set

    def get_subset_of_labeled_indices(self, indices: List[int]) -> Subset:
        assert all(i in self.labeled_indices for i in indices)
        return Subset(self.train_dataset, list(indices))

    def get_labeled_subset(self):
        """Get Subset for labeled data."""
        return Subset(self.train_dataset, list(self.labeled_indices))

    def get_unlabeled_subset(self):
        """Get Subset for unlabeled data."""
        return Subset(self.train_dataset, list(self.unlabeled_indices))

    def get_unlabeled_indices(self) -> List[int]:
        """Get list of unlabeled indices."""
        return list(self.unlabeled_indices)

    def get_labeled_indices(self) -> List[int]:
        """Get list of labeled indices."""
        return list(self.labeled_indices)

    def get_pool_stats(self) -> Dict[str, int]:
        """Get current pool statistics."""
        return {
            "labeled_count": len(self.labeled_indices),
            "unlabeled_count": len(self.unlabeled_indices),
            "total_count": len(self.train_dataset)
        }