import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch

import logging

from .config import ExperimentConfig
from .pool import DataPool

from .utils.data_loader import load_agnews
from .utils.text_dataset import TextDataset

class ActiveLearning:
    """Manages active learning round by round."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

        self.set_seeds()

        # Initializing stuff from cfg
        self.model = self._initialize_model()
        self.sampler = self.cfg.sampler
        self.strategy = self.cfg.strategy

        # Load data and initialize pool
        self.train_dataset, self.val_dataset, self.test_dataset = self._load_data() # Or just make it params in init...
        self._initialize_pool(self.train_dataset)

        # Round tracking
        self.round_stats: List[Dict] = []
        self.current_round = 0

    def _initialize_pool(self, train_dataset):
        # Initialize pool with random samples
        all_indices = list(range(len(train_dataset)))
        initial_indices = random.sample(all_indices, self.cfg.initial_pool_size)
        self.pool = DataPool(train_dataset, initial_indices)

    def _load_data(self):
        # TODO: Maybe we'll need to just make dataset as another param to init instead, idk, this is just a dummy
        # Make sure the index is reset, or change the pool initialization
        df_train, df_val, df_test = load_agnews(path='data')

        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

        # Create datasets
        train_dataset = TextDataset(
            texts=df_train['text'].tolist(),
            labels=df_train['label'].tolist()
        )

        val_dataset = TextDataset(
            texts=df_val['text'].tolist(),
            labels=df_val['label'].tolist()
        )

        test_dataset = TextDataset(
            texts=df_test['text'].tolist(),
            labels=df_test['label'].tolist()
        )

        return train_dataset, val_dataset, test_dataset

    def set_seeds(self):
        """Set all random seeds for reproducibility."""
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.cfg.seed)
            torch.cuda.manual_seed_all(self.cfg.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _initialize_model(self):
        """Sampler creation based on config."""
        if self.cfg.model_class:
            return self.cfg.model_class(**self.cfg.model_kwargs)
        else:
            raise ValueError("sampler_class must be specified in config.")

    def reset_model(self):
        """Reset model to initial state by recreating it."""
        logging.info("Resetting model to initial state . . .")
        self.model = self._initialize_model()

    def train_one_round(self, new_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Train one round of active learning.

        Args:
            new_indices: Indices of newly sampled data (None for first round)

        Returns:
            Dictionary with round statistics
        """
        logging.info(f"\n--- Round {self.current_round + 1}")

        if new_indices is not None and len(new_indices) > 0:
            logging.info(f"Training with {len(new_indices)} new samples")

        # Train model (timing handled in base class)
        self.model, training_stats = self.strategy.train(self.model, self.pool, new_indices)

        # Evaluate model
        f1_score = self._evaluate_model()

        # Compile round statistics
        # TODO: Add/remove if needed
        round_stats = {
            **training_stats,
            "f1_score": f1_score,
            "pool_stats": self.pool.get_pool_stats()
        }

        self.round_stats.append(round_stats)

        logging.info(f"Round {self.current_round + 1} complete: F1={f1_score:.4f}, "
                     f"Time={training_stats['training_time']:.2f}s")

        self.current_round += 1
        return round_stats

    def sample_next_batch(self, batch_size: Optional[int] = None) -> List[int]: # TODO: Maybe removing the batch size from here (only use cfg provided)
        """
        Sample next batch of data to label.

        Returns:
            List of selected indices
        """
        if batch_size is None:
            batch_size = self.cfg.batch_size
        elif self.cfg.batch_size and batch_size != self.cfg.batch_size:
            logging.warning(f"Using provided batch size {batch_size} instead of {self.cfg.batch_size} in config.")

        if not self.pool.get_unlabeled_indices():
            logging.warning("No unlabeled data remaining!")
            return []

        selected_indices = self.sampler.select(self.pool, self.model, batch_size)
        # DO NOT ADD THEM TO THE POOL RIGHT AWAY DUMBAHH
        # Letting the strategy decide what to do with it
        # self.pool.add_labeled_samples(selected_indices)

        logging.info(f"Sampled {len(selected_indices)} new samples using {type(self.sampler).__name__}")
        return selected_indices

    def run_full_pipeline(self):
        """Running full pipeline of active learning for provided amount of rounds."""
        # TODO: If needed we could just run the whole active learning thing here . . .
        pass


    def _evaluate_model(self) -> float:
        """Evaluate model and return macro F1 score."""
        # TODO: Implement actual evaluation logic (or we could do it outside of this class and remove it from stats)
        return random.random()  # Dummy score for now

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all rounds."""
        return {
            "cfg": self.cfg,
            "total_rounds": len(self.round_stats),
            "round_stats": self.round_stats,
            "final_pool_stats": self.pool.get_pool_stats(),
            "final_f1": self.round_stats[-1]["f1_score"] if self.round_stats else 0.0
        }

    def save_experiment(self, filepath: Path = None) -> None:
        """Save experiment results to file."""
        # TODO: needs better usable/readable saving format (since stuff like scheduler, optimizer is in strategy class
        if filepath is None:
            save_dir = self.cfg.save_dir / self.cfg.experiment_name
            save_dir.mkdir(parents=True, exist_ok=True)
            filepath = save_dir / f"round_{self.current_round}_results.json"

        summary = self.get_experiment_summary()

        # Converting non-serializable objects
        summary["cfg"] = summary["cfg"].__dict__

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logging.info(f"Experiment saved to {filepath}")

    def reset_experiment(self):
        """Reset entire experiment to initial state."""
        # TODO: not sure if will be needed
        self.reset_model()
        self._initialize_pool(self.train_dataset)
        self.round_stats = []
        self.current_round = 0
        logging.info("Experiment reset to initial state")