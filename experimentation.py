import itertools

import torch
from pathlib import Path
import json

from adaptive_al_v2.active_learning import ActiveLearning, ExperimentConfig

import logging

if __name__ == "__main__":
    # Comment it out if you dont want to see info logs
    logging.basicConfig(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    data_sets_num_labels = {"agnews": 4, "imdb": 2, "jigsaw": 2}
    seeds = [42, 43, 44, 45, 46]
    strategies = ["DeltaF1Strategy", "FineTuneStrategy", "NewOnlyStrategy", "RetrainStrategy"]
    data_sets = ["agnews", "imdb", "jigsaw"]
    common_config_parameters = {

        "save_dir": Path("./experiments"),

        # Pool settings
        "initial_pool_size": 200,
        "acquisition_batch_size": 32,

        # Plateau checking:
        "min_rounds_before_plateau": 10,
        "plateau_patience": 10,
        "plateau_f1_threshold": 0.001,
        "pool_proportion_threshold": 0.2,

        # Model
        "model_name_or_path": "distilbert-base-uncased",
        "tokenizer_kwargs": {
            "max_length": 128,
            "padding": "max_length",
            "truncation": True,
            "add_special_tokens": True,
            "return_tensors": "pt"
        },

        "optimizer_class": "Adam",
        "optimizer_kwargs": {"lr": 2e-5, "weight_decay": 1e-3},

        "criterion_class": "CrossEntropyLoss",
        "criterion_kwargs": {},

        "scheduler_class": "StepLR",
        "scheduler_kwargs": {"step_size": 10, "gamma": 0.1},

        # Sampler
        "sampler_class": "EntropyOnRandomSubsetSampler",
        "sampler_kwargs": {"random_subset_size": 5000},

        # Training
        "device": device,
        "epochs": 5,
        "batch_size": 16
    }

    switch_epsilon = [0.1, 0.05, 0.01, 0.005]
    switch_k = [2, 3, 6]

    uncommon_config_parameters_instances = itertools.product(strategies, data_sets, seeds)

    for strategy, data_set, seed in uncommon_config_parameters_instances:
        strategy_kwargs_instances = [{'epsilon':e, 'k':k} for k, e in itertools.product(switch_k, switch_epsilon)]\
            if strategy == "DeltaF1Strategy" else [{}]
        for strat_kwargs in strategy_kwargs_instances:
            experiment_name =  f"{strategy}_{'_'.join(strat_kwargs)}_{data_set}_{seed}"
            config_parameters = common_config_parameters.copy()

            config_parameters.update({"experiment_name": experiment_name,
                                      "data": data_set,
                                      "seed": seed,
                                      "num_labels": data_sets_num_labels[data_set],
                                      "strategy_class": strategy,
                                      "strategy_kwargs": strat_kwargs
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
