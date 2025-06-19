from typing import List, Tuple


class ActiveLearningPool:
    """
    Manages the labeled and unlabeled pools for active learning.
    """

    def __init__(self, dataset, sampler, model):
        self.dataset = dataset
        self.unlabeled = set(range(len(dataset)))
        self.labeled = set()
        self.last_round = set()
        self.sampler = sampler
        self.model = model

    def query(self, k: int) -> List[int]:
        """
        Sample k new examples from the unlabeled pool using the provided sampler and model.
        Updates internal labeled/unlabeled sets and tracks the last round's selection.
        """
        pass

    def get_full_loader(self, batch_size: int, shuffle: bool = True) -> Tuple:
        """
        Returns dataloader over all labeled data so far
          - new_loader: over only the examples added in the last query
        """
        pass

    def get_new_loader(self, batch_size: int, shuffle: bool = True) -> Tuple:
        """
        Returns dataloader over only the examples added in the last query
        """
        pass

    def reset(self):
        """
        Clear all labeled and last_round data
        """
        self.unlabeled = set(range(len(self.dataset)))
        self.labeled.clear()
        self.last_round.clear()
