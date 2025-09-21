import random

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
    def __init__(self, epsilon: float, k: int, validation_fraction:float, **kwargs, ):
        super().__init__(**kwargs)

        self.epsilon = epsilon
        self.k = k
        self.validation_indices = []
        self.validation_fraction = validation_fraction
        self.count = 0
        self.switched = False
        self.prev_f1 = None

        self.fine_tune = FineTuneStrategy(strategy=self)
        self.retrain = RetrainStrategy(strategy=self)

    def _calc_f1(self, subset_to_evaluate):
        stats = evaluate_model(self.model, self.criterion, self.batch_size, subset_to_evaluate, self.device)
        return stats['f1_score']

    def _train_implementation(self, pool: DataPool, new_indices: List[int]) -> Dict:
        if self.switched:
            return self.fine_tune._train_implementation(pool, new_indices)

        new_validation_indices = new_indices[int(self.validation_fraction * len(new_indices)):]
        self.validation_indices.extend(new_validation_indices)
        new_train_indices = new_indices[:len(new_indices) - len(new_validation_indices)]

        stats = self.retrain._train_implementation(pool, new_train_indices)

        cur_f1 = self._calc_f1(pool.get_subset(self.validation_indices))
        # cur_f1 = stats['f1_score']
        delta_f1 = cur_f1 - self.prev_f1 if self.prev_f1 is not None else float('inf') # We don't count the first round right?
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