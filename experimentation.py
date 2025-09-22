import itertools

import torch
from pathlib import Path
import json

from adaptive_al.active_learning import ActiveLearning, ExperimentConfig

import logging

if __name__ == "__main__":
    # Comment it out if you dont want to see info logs
    logging.basicConfig(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    data_sets_num_labels = {"agnews": 4, "imdb": 2, "jigsaw": 2}
    seeds = [42, 43, 44,]
    strategies = ["DeltaF1Strategy", "FineTuneStrategy", "NewOnlyStrategy", "RetrainStrategy", ]
    data_sets = ["agnews", "imdb", "jigsaw"]
    common_config_parameters = {

        "save_dir": Path("./results/sep_22"),

        # Pool settings
        "initial_pool_size": 200,
        "acquisition_batch_size": 32,
        "max_seconds": 3600,
        "total_rounds": 50,

        # Plateau checking:
        "min_rounds_before_plateau": 10,
        "plateau_patience": 10,
        "plateau_f1_threshold": 0.001,

        # Model
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

        # Sampler
        "sampler_class": "EntropyOnRandomSubsetSampler",

        # Training
        "device": device,
        "epochs": 5,
        "batch_size": 16
    }

    switch_epsilon = [0.05, 0.01, 0.005]
    switch_k = [3, 5, 7]
    validation_fraction = [0.05, 0.1, 0.25]
    uncommon_config_parameters_instances = itertools.product(strategies, data_sets, seeds)

    for strategy, data_set, seed in uncommon_config_parameters_instances:
        strategy_kwargs_instances = [{'validation_fraction': v, 'epsilon': e, 'k': k}
                                     for v, k, e in itertools.product(validation_fraction, switch_k, switch_epsilon)] \
            if strategy == "DeltaF1Strategy" else [{}]
        for strat_kwargs in strategy_kwargs_instances:
            experiment_name = f"{strategy}_{'_'.join([str(v).replace('.', '') for v in strat_kwargs.values()])}_{data_set}_{seed}"
            config_parameters = common_config_parameters.copy()
            config_parameters.update({"experiment_name": experiment_name,
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
            logging.info(
                "Final Test Metrics: F1=%.4f, Accuracy=%.4f, Loss=%.4f",
                final_metrics['f1_score'],
                final_metrics['accuracy'],
                final_metrics['loss']
            )
            al.save_experiment()
