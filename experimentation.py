"""
Hyperparameter optimization and experiment runner for active learning strategies.

This script runs active learning experiments across multiple datasets, seeds, and
strategies. For DeltaF1Strategy, it uses Optuna to optimize hyperparameters.
For other strategies, it runs experiments directly.

Usage:
    # Run with defaults
    python experimentation.py

    # Override specific parameters
    python experimentation.py --datasets agnews --epochs 10 --learning-rate 3e-5

    # See all options
    python experimentation.py --help
"""

import itertools
import time
import argparse

import torch
from pathlib import Path
import logging
import optuna

from adaptive_al.active_learning import ActiveLearning, ExperimentConfig
HOURS = 3600

def objective(trial, strategy, data_sets, seeds, common_config_parameters, data_sets_num_labels, start_time=0):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.trial.Trial): Current Optuna trial object.
        strategy (str): Active learning strategy name.
        data_sets (list): List of dataset names to evaluate on.
        seeds (list): Random seeds for reproducibility.
        common_config_parameters (dict): Base configuration dictionary shared across experiments.
        data_sets_num_labels (dict): Mapping of dataset names to number of labels.
        start_time (float): Start time of the experiment (for logging elapsed time).

    Returns:
        float: Average F1 score across datasets and seeds for the given trial.
    """
    # Hyperparameters chosen by Optuna
    validation_fraction = trial.suggest_categorical("validation_fraction", [0.02, 0.05, 0.1, 0.2])
    epsilon = trial.suggest_categorical("epsilon", [0.05, 0.02, 0.01, 0.005])
    k = trial.suggest_categorical("k", [4, 6, 9, 13])

    strat_kwargs = {
        "validation_fraction": validation_fraction,
        "epsilon": epsilon,
        "k": k
    }

    f1_scores = []
    # Loop through datasets and seeds
    for data_set in data_sets:
        for seed in seeds:
            logging.info(f"Evaluating hyperparameters: validation_fraction: {validation_fraction}, epsilon: {epsilon}, k: {k}")
            logging.info(f"Starting experiment {strategy} on {data_set} with seed {seed}.")
            logging.info(f"Time passed: {round((time.perf_counter() - start_time)/HOURS, 2)} hours.")
            experiment_name = f"{strategy}_{validation_fraction}_{epsilon}_{k}_{data_set}_{seed}"

            config_parameters = common_config_parameters.copy()
            config_parameters.update({
                "experiment_name": experiment_name,
                "data": data_set,
                "seed": seed,
                "num_labels": data_sets_num_labels[data_set],
                "strategy_class": strategy,
                "strategy_kwargs": strat_kwargs,
                "sampler_kwargs": {**common_config_parameters["sampler_kwargs"], "seed": seed}
            })

            experiment_config = ExperimentConfig(**config_parameters)
            al = ActiveLearning(experiment_config)

            final_metrics = al.run_full_pipeline()
            al.save_experiment()

            f1_scores.append(final_metrics["f1_score"])

    if len(f1_scores) == 0:
        return None

    avg_f1 = sum(f1_scores) / len(f1_scores)
    return avg_f1


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments with defaults.
    """
    parser = argparse.ArgumentParser(
        description="Run active learning experiments with optional hyperparameter optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Experiment selection
    parser.add_argument("--datasets", nargs="+", default=["agnews", "imdb", "jigsaw"],
                        help="Datasets to run experiments on")
    parser.add_argument("--strategies", nargs="+",
                        default=["DeltaF1Strategy", "FineTuneStrategy", "NewOnlyStrategy", "RetrainStrategy"],
                        help="Strategies to evaluate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                        help="Random seeds for reproducibility")

    # Output
    parser.add_argument("--save-dir", type=str, default="./experiments/sep_25",
                        help="Directory to save results")

    # Model settings
    parser.add_argument("--model", type=str, default="distilbert-base-uncased",
                        help="Hugging Face model name or path")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Maximum tokenization length")

    # Training settings
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs per round")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3,
                        help="Weight decay")

    # Active learning settings
    parser.add_argument("--initial-pool-size", type=int, default=200,
                        help="Initial labeled pool size")
    parser.add_argument("--acquisition-batch-size", type=int, default=32,
                        help="Number of samples to acquire per round")
    parser.add_argument("--total-rounds", type=int, default=40,
                        help="Maximum number of active learning rounds")
    parser.add_argument("--max-seconds", type=int, default=2400,
                        help="Maximum seconds per experiment")

    # Plateau detection
    parser.add_argument("--min-rounds-before-plateau", type=int, default=10,
                        help="Minimum rounds before checking for plateau")
    parser.add_argument("--plateau-patience", type=int, default=10,
                        help="Rounds of no improvement before stopping")
    parser.add_argument("--plateau-f1-threshold", type=float, default=0.0005,
                        help="Minimum F1 improvement to avoid plateau detection")

    # Sampling
    parser.add_argument("--random-subset-size", type=int, default=1000,
                        help="Random subset size for entropy sampler")

    # Scheduler
    parser.add_argument("--scheduler-step-size", type=int, default=10,
                        help="Step size for StepLR scheduler")
    parser.add_argument("--scheduler-gamma", type=float, default=0.1,
                        help="Gamma for StepLR scheduler")

    # Optuna
    parser.add_argument("--optuna-hours", type=float, default=15,
                        help="Maximum hours for Optuna optimization")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    start_time = time.perf_counter()
    logging.basicConfig(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Number of labels per dataset
    data_sets_num_labels = {"agnews": 4, "imdb": 2, "jigsaw": 2}

    # Seeds, strats and datasets from CLI
    seeds = args.seeds
    strategies = args.strategies
    data_sets = args.datasets

    # Common configuration parameters
    # Note: Optimizer (Adam), loss function (CrossEntropyLoss), scheduler (StepLR),
    # and sampler (EntropyOnRandomSubsetSampler) are fixed.
    common_config_parameters = {
        "save_dir": Path(args.save_dir),
        "initial_pool_size": args.initial_pool_size,
        "acquisition_batch_size": args.acquisition_batch_size,
        "max_seconds": args.max_seconds,
        "total_rounds": args.total_rounds,
        "min_rounds_before_plateau": args.min_rounds_before_plateau,
        "plateau_patience": args.plateau_patience,
        "plateau_f1_threshold": args.plateau_f1_threshold,
        "model_name_or_path": args.model,
        "tokenizer_kwargs": {
            "max_length": args.max_length,
            "padding": "max_length",
            "truncation": True,
            "add_special_tokens": True,
            "return_tensors": "pt"
        },
        "optimizer_class": "Adam",
        "optimizer_kwargs": {"lr": args.learning_rate, "weight_decay": args.weight_decay},
        "criterion_class": "CrossEntropyLoss",
        "criterion_kwargs": {},
        "scheduler_class": "StepLR",
        "scheduler_kwargs": {"step_size": args.scheduler_step_size, "gamma": args.scheduler_gamma},
        "sampler_class": "EntropyOnRandomSubsetSampler",
        "sampler_kwargs": {"random_subset_size": args.random_subset_size},
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }

    # Print configuration
    print("=" * 80)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 80)
    for key, value in common_config_parameters.items():
        print(f"{key}: {value}")
    print(f"datasets: {data_sets}")
    print(f"strategies: {strategies}")
    print(f"seeds: {seeds}")
    print("=" * 80)

    # Run experiments for each strategy
    for strategy in strategies:
        if strategy == "DeltaF1Strategy":
            logging.info(f"Running Optuna study for {strategy} (averaging across datasets & seeds)")
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: objective(trial, strategy, data_sets, seeds, common_config_parameters,
                                        data_sets_num_labels, start_time),
                timeout=int(args.optuna_hours * HOURS)
            )
            logging.info(f"Best params for {strategy}: {study.best_params}, Avg F1={study.best_value:.4f}")
        else:
            # For other strategies, just run experiments directly without Optuna
            for data_set, seed in itertools.product(data_sets, seeds):
                experiment_name = f"{strategy}_{data_set}_{seed}"
                experiment_path = common_config_parameters["save_dir"] / experiment_name
                if experiment_path.exists():
                    logging.info(f"Skipping existing experiment: {experiment_name}")
                    continue

                # Build configuration for this experiment
                config_parameters = common_config_parameters.copy()
                config_parameters.update({
                    "experiment_name": experiment_name,
                    "data": data_set,
                    "seed": seed,
                    "num_labels": data_sets_num_labels[data_set],
                    "strategy_class": strategy,
                    "strategy_kwargs": {},
                    "sampler_kwargs": {**common_config_parameters["sampler_kwargs"], "seed": seed}
                })

                # Run active learning pipeline
                experiment_config = ExperimentConfig(**config_parameters)
                al = ActiveLearning(experiment_config)
                final_metrics = al.run_full_pipeline()
                al.save_experiment()
                logging.info(
                    "Final Test Metrics (%s): F1=%.4f, Accuracy=%.4f, Loss=%.4f",
                    experiment_name,
                    final_metrics['f1_score'],
                    final_metrics['accuracy'],
                    final_metrics['loss']
                )

    logging.info("Experimentation Complete")
