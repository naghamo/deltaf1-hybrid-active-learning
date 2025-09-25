import itertools
import time

import torch
from pathlib import Path
import logging
import optuna

from adaptive_al.active_learning import ActiveLearning, ExperimentConfig
HOURS = 3600
def objective(trial, strategy, data_sets, seeds, common_config_parameters, data_sets_num_labels, start_time=0):
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
                "sampler_kwargs": {"random_subset_size": 1000, "seed": seed}
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

if __name__ == "__main__":
    start_time = time.perf_counter()
    logging.basicConfig(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_sets_num_labels = {"agnews": 4, "imdb": 2, "jigsaw": 2}
    seeds = [42, 43, 44]
    strategies = ["DeltaF1Strategy", "FineTuneStrategy", "NewOnlyStrategy", "RetrainStrategy"]
    data_sets = ["agnews", "imdb", "jigsaw"]

    common_config_parameters = {
        "save_dir": Path("./results/sep_25"),
        "initial_pool_size": 200,
        "acquisition_batch_size": 32,
        "max_seconds": 2400,
        "total_rounds": 40,
        "min_rounds_before_plateau": 10,
        "plateau_patience": 10,
        "plateau_f1_threshold": 0.0005,
        "model_name_or_path": "distilbert-base-uncased",
        "tokenizer_kwargs": {
            "max_length": 128,
            "padding": "max_length",
            "truncation": True,
            "add_special_tokens": True,
            "return_tensors": "pt"
        },
        "approximate_evaluation_subset_size": 6000,
        "optimizer_class": "Adam",
        "optimizer_kwargs": {"lr": 2e-5, "weight_decay": 1e-3},
        "criterion_class": "CrossEntropyLoss",
        "criterion_kwargs": {},
        "scheduler_class": "StepLR",
        "scheduler_kwargs": {"step_size": 10, "gamma": 0.1},
        "sampler_class": "EntropyOnRandomSubsetSampler",
        "device": device,
        "epochs": 5,
        "batch_size": 16
    }

    for strategy in strategies:
        if strategy == "DeltaF1Strategy":
            logging.info(f"Running Optuna study for {strategy} (averaging across datasets & seeds)")
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: objective(trial, strategy, data_sets, seeds, common_config_parameters, data_sets_num_labels, start_time),
                timeout=15*HOURS
            )
            logging.info(f"Best params for {strategy}: {study.best_params}, Avg F1={study.best_value:.4f}")
        else:
            for data_set, seed in itertools.product(data_sets, seeds):
                experiment_name = f"{strategy}_{data_set}_{seed}"
                experiment_path = common_config_parameters["save_dir"] / experiment_name
                if experiment_path.exists():
                    logging.info(f"Skipping existing experiment: {experiment_name}")
                    continue

                config_parameters = common_config_parameters.copy()
                config_parameters.update({
                    "experiment_name": experiment_name,
                    "data": data_set,
                    "seed": seed,
                    "num_labels": data_sets_num_labels[data_set],
                    "strategy_class": strategy,
                    "strategy_kwargs": {},
                    "sampler_kwargs": {"random_subset_size": 1000, "seed": seed}
                })

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
