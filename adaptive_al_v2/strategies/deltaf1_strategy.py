import torch
from sympy.physics.paulialgebra import epsilon
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple

import logging

from .base_strategy import BaseStrategy
from ..pool import DataPool

from ..evaluation import evaluate_model

from .fine_tuning_strategy import FineTuneStrategy
from .retrain_strategy import RetrainStrategy

class DeltaF1Strategy(BaseStrategy):
    def __init__(self, epsilon: float, k: int, validation_fraction: float = 0, **kwargs):
        super().__init__(**kwargs)

        # Or anything else specific to the deltaf1 method . . . then pass those in kwargs of cfg
        self.epsilon = epsilon
        self.k = k

        self.count = 0
        self.switched = False
        self.prev_f1 = None
        self.validation_indices = []
        self.validation_fraction = validation_fraction

        self.fine_tune = FineTuneStrategy(strategy=self)
        self.retrain = RetrainStrategy(strategy=self)

    def pass_args_to_sampler(self) -> Dict[str, Any]:
        return {"random_indices_fraction": 0 if self.switched else self.validation_fraction}

    def _calc_f1(self, subset_to_evaluate):
        stats = evaluate_model(self.model, self.criterion, self.batch_size, subset_to_evaluate, self.device)
        return stats['f1_score']

    def _separate_validation_and_train(self, new_indices: List[int]):
        new_validation_size = int(len(new_indices)*self.validation_fraction)
        new_train_size = len(new_indices) - new_validation_size
        return new_indices[:new_validation_size], new_indices[new_train_size:]

    def _train_implementation(self, pool: DataPool, new_indices: List[int]) -> Dict:
        if self.switched:
            return self.fine_tune._train_implementation(pool, new_indices)

        new_validation_indices, new_train_indices = self._separate_validation_and_train(new_indices)
        self.validation_indices.extend(new_validation_indices)
        stats = self.retrain._train_implementation(pool, new_train_indices)

        # On what to evaluate???
        cur_f1 = self._calc_f1(pool.get_subset(self.validation_indices))
        # cur_f1 = stats['f1_score']
        delta_f1 = cur_f1 - self.prev_f1 if self.prev_f1 is not None else float('inf') # We dont count the first round right?
        self.prev_f1 = cur_f1

        if abs(delta_f1) < self.epsilon:
            self.count += 1
        else:
            self.count = 0

        if self.count >= self.k and not self.switched:

            logging.info(f'Switched to finetune after {self.count} consecutive rounds <{self.epsilon}. Final delta_f1: {delta_f1}')
            self.switched = True
        else:
            logging.info(f'delta_f1: {delta_f1}, {self.count} consecutive rounds.')

        return stats