"""
Delta F1 adaptive training strategy.

This module implements our proposed hybrid strategy that adaptively switches from full retraining
to fine-tuning based on validation F1 score improvements. When F1 improvements
fall below a threshold for k consecutive rounds, the strategy switches to
fine-tuning for efficiency.
"""

from typing import List, Dict, Any, Tuple

import logging

from .base_strategy import BaseStrategy
from ..pool import DataPool

from ..evaluation import evaluate_model

from .fine_tuning_strategy import FineTuneStrategy
from .retrain_strategy import RetrainStrategy

class DeltaF1Strategy(BaseStrategy):
    """
    Adaptive strategy that switches from retraining to fine-tuning based on F1 improvement.

    This strategy monitors F1 score changes on a validation subset. When the absolute
    change in F1 score remains below epsilon for k consecutive rounds, it switches
    from full retraining to incremental fine-tuning for improved efficiency.

    Attributes:
        epsilon (float): F1 change threshold for switching to fine-tuning.
        k (int): Number of consecutive rounds below epsilon before switching.
        validation_fraction (float): Fraction of new samples to use for validation monitoring.
        count (int): Current count of consecutive rounds below epsilon threshold.
        switched (bool): Whether strategy has switched to fine-tuning mode.
        prev_f1 (float): Previous round's F1 score on validation set.
        validation_indices (List[int]): Indices used for validation monitoring.
        fine_tune (FineTuneStrategy): Fine-tuning strategy instance.
        retrain (RetrainStrategy): Retraining strategy instance.
    """
    def __init__(self, epsilon: float, k: int, validation_fraction: float = 0, **kwargs):
        """
        Initialize the Delta F1 adaptive strategy.

        Args:
            epsilon (float): Threshold for absolute F1 change. When |ΔF1| < epsilon
                            for k consecutive rounds, switch to fine-tuning.
            k (int): Number of consecutive rounds below epsilon before switching.
            validation_fraction (float): Fraction of each new batch to reserve for
                                        validation monitoring (default: 0).
            **kwargs: Additional arguments passed to BaseStrategy (model, optimizer, etc.).
        """
        super().__init__(**kwargs)

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
        """
        Provide validation fraction to sampler for random selection.

        Returns:
            Dict[str, Any]: Dictionary with 'random_indices_fraction' set to
                           validation_fraction before switch, 0 after switch.
        """
        return {"random_indices_fraction": 0 if self.switched else self.validation_fraction}

    def _calc_f1(self, subset_to_evaluate):
        """
        Calculate F1 score on a given subset.

        Args:
            subset_to_evaluate: Dataset subset to evaluate on.

        Returns:
            float: Macro F1 score on the subset.
        """
        stats = evaluate_model(self.model, self.criterion, self.batch_size, subset_to_evaluate, self.device)
        return stats['f1_score']

    def _separate_validation_and_train(self, new_indices: List[int]):
        """
        Split new indices into validation and training sets.

        Allocates the first validation_fraction of indices for validation monitoring,
        with the remainder for training.

        Args:
            new_indices (List[int]): Newly sampled indices to split.

        Returns:
            tuple: (validation_indices, training_indices). Returns ([], None) if
                   new_indices is empty.
        """
        if not new_indices:
            return [], None
        else:
            new_validation_size = int(len(new_indices)*self.validation_fraction)
            new_train_size = len(new_indices) - new_validation_size
            return new_indices[:new_validation_size], new_indices[new_train_size:]

    def _train_implementation(self, pool: DataPool, new_indices: List[int]) -> Dict:
        """
        Train using retrain or fine-tune strategy based on F1 improvement tracking.

        If switched to fine-tuning mode, delegates to FineTuneStrategy. Otherwise,
        separates new indices into validation and training sets, trains using
        RetrainStrategy, evaluates F1 on validation set, and tracks consecutive
        rounds below epsilon threshold.

        Args:
            pool (DataPool): Current data pool with labeled/unlabeled splits.
            new_indices (List[int]): Newly sampled indices (not yet in pool).

        Returns:
            Dict: Training statistics from the underlying strategy.
        """
        if self.switched:
            return self.fine_tune._train_implementation(pool, new_indices)

        new_validation_indices, new_train_indices = self._separate_validation_and_train(new_indices)
        self.validation_indices.extend(new_validation_indices)
        stats = self.retrain._train_implementation(pool, new_train_indices)

        if not self.validation_indices:
            return stats

        pool.add_labeled_samples(new_validation_indices)
        cur_f1 = self._calc_f1(pool.get_subset(self.validation_indices))
        delta_f1 = cur_f1 - self.prev_f1 if self.prev_f1 is not None else float('inf')
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
