from abc import ABC, abstractmethod
from typing import Dict, List


class BaseStrategy(ABC):
    """
    Abstract base class for active learning training strategies.
    """
    def __init__(self, model, model_class, model_kwargs, pool):
        self.model = model
        self.model_class = model_class
        self.model_kwargs = model_kwargs

        self.pool = pool

    @abstractmethod
    def train_round(self, new_indices: List[int], round_i: int) -> None:
        """
        new_indices: the *new* examples sampled this round
        round_i:      the round number (0, 1, 2 ...)
        """
        pass

    def reset_model(self):
        # there is probably a better way to do it
        self.model = self.model_class(**self.model_kwargs)