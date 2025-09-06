from datetime import datetime
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch

from .utils.text_datasets import TextClassificationDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import torch.nn as nn
import torch.optim as optim

import logging

from .config import ExperimentConfig
from .pool import DataPool

import adaptive_al_v2.strategies as strategies
import adaptive_al_v2.samplers as samplers

from torch.utils.data import DataLoader

from .utils.data_loader import load_agnews
from .evaluation import evaluate_model


class ActiveLearning:
    """Manages active learning round by round."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self._initialize()

    def _initialize(self):
        """Initialize everything for the active learning rounds."""
        self.set_seeds(self.cfg.seed)

        self._initialize_model_and_tokenizer()

        self._initialize_data()
        self._initialize_pool()

        self._initialize_classes()
        self._initialize_round_tracking()

    def _initialize_model_and_tokenizer(self):
        """Loads the model and tokenizer from Hugging Face."""
        logging.info(f"Loading tokenizer and model from '{self.cfg.model_name_or_path}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_name_or_path,
            num_labels=self.cfg.num_labels
        )

    def _initialize_data(self):
        """Initialize datasets"""
        self.train_dataset, self.val_dataset, self.test_dataset = self._load_data()
        logging.info(
            f"Train size: {len(self.train_dataset)}, Validation size: {len(self.val_dataset)}, Test size: {len(self.test_dataset)}")

    def _initialize_pool(self):
        """
        Initialize the pool for active learning rounds.
        """
        # Initialize pool with random samples
        all_indices = list(range(len(self.train_dataset)))
        initial_indices = random.sample(all_indices, self.cfg.initial_pool_size)
        self.pool = DataPool(self.train_dataset, self.val_dataset, self.test_dataset, initial_indices)

    def _initialize_classes(self):
        """
        Resolve the provided class names. Each one needs to be properly stored in corresponding module.
        """
        cfg = self.cfg

        # --- Optimizer
        self.optimizer_cls = resolve_class(cfg.optimizer_class, optim)  # from torch.optim
        self.optimizer_kwargs = cfg.optimizer_kwargs

        # --- Criterion
        self.criterion_cls = resolve_class(cfg.criterion_class, nn)  # from torch.nn
        self.criterion_kwargs = cfg.criterion_kwargs

        # --- Scheduler
        self.scheduler_cls = None
        self.scheduler_kwargs = {}
        if cfg.scheduler_class:
            self.scheduler_cls = resolve_class(cfg.scheduler_class, optim.lr_scheduler)  # from torch.optim.lr_scheduler
            self.scheduler_kwargs = cfg.scheduler_kwargs

        # --- Strategy
        self.strategy_cls = resolve_class(cfg.strategy_class, strategies)  # from our adaptive_al.strategies
        self.strategy_kwargs = cfg.strategy_kwargs

        # --- Sampler
        self.sampler_cls = resolve_class(cfg.sampler_class, samplers)  # from our adaptive_al.samplers
        self.sampler_kwargs = cfg.sampler_kwargs

        # --- Instantiate strategy
        # if cfg.strategy_class == 'DeltaF1Strategy':
        #     self.strategy_kwargs['val_dataset'] = self.val_dataset

        self.strategy = self.strategy_cls(
            model=self.model,  # Passing model instance
            optimizer_cls=self.optimizer_cls,
            optimizer_kwargs=self.optimizer_kwargs,
            criterion_cls=self.criterion_cls,
            criterion_kwargs=self.criterion_kwargs,
            scheduler_cls=self.scheduler_cls,
            scheduler_kwargs=self.scheduler_kwargs,
            device=cfg.device,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            # pool: DataPool = self.pool,
            # val_dataset=self.val_dataset,
            **self.strategy_kwargs  # optional extra kwargs
        )

        # --- Instantiate sampler
        self.sampler = self.sampler_cls(model=self.model, batch_size=cfg.batch_size, device=cfg.device,
                                        **self.sampler_kwargs)

    def _initialize_round_tracking(self):
        """
        Initialize properties to start round tracking.
        """
        self.round_stats: List[Dict] = []
        self.final_test_stats: Dict = {}
        self.current_round = 0

    def _load_data(self):
        dataset_name = self.cfg.data

        # Make sure the index is reset, or we need to change the pool initialization
        train_dataset, val_dataset, test_dataset = eval(
            f"load_{dataset_name}(path='data', seed={self.cfg.seed}, model_name_or_path='{self.cfg.model_name_or_path}', tokenizer_kwargs={self.cfg.tokenizer_kwargs})")

        return train_dataset, val_dataset, test_dataset

    def set_seeds(self, seed):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
        training_stats = self.strategy.train(self.pool, new_indices)

        # Evaluate model
        val_stats = evaluate_model(self.model, self.strategy.criterion, self.cfg.batch_size, dataset=self.val_dataset,
                                   device=self.cfg.device)

        # Compile round statistics
        # TODO: Add/remove if needed
        round_stats = {
            **training_stats,
            **val_stats,
            "pool_stats": self.pool.get_pool_stats()
        }

        self.round_stats.append(round_stats)
        logging.info(
            f"Round {self.current_round + 1} complete. Val Stats: Loss={val_stats['loss']}, F1={val_stats['f1_score']}, "
            f"Time={training_stats['training_time']:.2f}s")

        self.current_round += 1
        return round_stats

    def sample_next_batch(self) -> List[int]:
        """
        Sample next batch of data to label.

        Returns:
            List of selected indices
        """
        if not self.pool.get_unlabeled_indices():
            logging.warning("No unlabeled data remaining!")
            return []

        selected_indices = self.sampler.select(self.pool, self.cfg.acquisition_batch_size)
        # DO NOT ADD THEM TO THE POOL RIGHT AWAY DUMBAHH
        # Letting the strategy decide what to do with it
        # self.pool.add_labeled_samples(selected_indices)

        logging.info(f"Sampled {len(selected_indices)} new samples using {type(self.sampler).__name__}")
        return selected_indices

    def run_full_pipeline(self):
        """Running full pipeline of active learning for provided amount of rounds."""
        self._initialize()
        total_rounds = self.cfg.total_rounds

        if total_rounds == -1:
            total_rounds = float('inf')
            logging.info(f"Running rounds until we all available data is used.")
        else:
            logging.info(f"Running {total_rounds} rounds.")

        new_indices = []
        if total_rounds > 0:
            self.train_one_round(None)
            new_indices = self.sample_next_batch()

        while new_indices and self.current_round < total_rounds:
            self.train_one_round(new_indices)
            new_indices = self.sample_next_batch()

        if self.current_round != total_rounds:
            logging.info(f"\n--- Stopped at {self.current_round} (ran out of samples).")

        metrics = evaluate_model(self.model, self.strategy.criterion, self.cfg.batch_size, dataset=self.test_dataset,
                                 device=self.cfg.device)
        self.final_test_stats = metrics

        logging.info(
            f"Final Test set evaluation: Loss={metrics['loss']}, F1={metrics['f1_score']:.4f}, Acc={metrics['accuracy']:.4f}")
        return metrics

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all rounds."""
        return {
            "cfg": self.cfg.__dict__,  # Already fully serializable
            "total_rounds": len(self.round_stats),
            "round_val_stats": self.round_stats,
            "final_pool_stats": self.pool.get_pool_stats(),
            "final_test_stats": self.final_test_stats
        }

    def save_experiment(self, filepath: Optional[Path] = None, timestamp: bool = True) -> None:
        """Save experiment results to a JSON file."""
        summary = self.get_experiment_summary()

        # Determine file path
        save_dir = self.cfg.save_dir / self.cfg.experiment_name
        save_dir.mkdir(parents=True, exist_ok=True)

        if filepath is None:
            name = f"results"
            if timestamp:
                name += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = save_dir / f"{name}.json"

        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)  # default=str handles any residual non-serializable objects

        logging.info(f"Experiment saved to {filepath}")


def resolve_class(name: str, module) -> Any:
    """
    Resolve a class name (string) to a Python class.

    Args:
        name: Name of the class (string)
        module: Module to look in (e.g., torch.nn, torch.optim)
    Returns:
        Python class object
    """
    if hasattr(module, name):
        return getattr(module, name)

    raise ValueError(f"Cannot resolve class '{name}' in module {module}")
