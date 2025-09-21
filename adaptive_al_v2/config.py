from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     pass


@dataclass
class ExperimentConfig:
    # --- Reproducibility
    # seeds: List[int] = field(default_factory=lambda: [9, 31, 8,
    #                                                   106, 7,
    #                                                   207, 15]) # Seeds for multiple runs to average results
    seed: int = 42
    total_rounds: int = -1 # Total rounds to run active learning (model training + sampling new data), -1 for until we run out of data

    # --- Pool config
    initial_pool_size: int = 200 # Initial size of the pool for training
    acquisition_batch_size: int = 32 # New labels per round

    # Plateau checking:
    min_rounds_before_plateau: int = -1
    plateau_patience: int = -1
    plateau_f1_threshold: float = 0.5

    approximate_evaluation_subset_size: int = -1

    pool_proportion_threshold: float = 1

    sampler_class: str = field(default=None)
    sampler_kwargs: Dict[str, Any] = field(default_factory=dict)

    # --- Training config
    # Strategy and Sampler configuration
    strategy_class: str = field(default=None)
    strategy_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Model configuration (via transformers)
    model_name_or_path: str = field(default='None')
    num_labels: int = field(default=None)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "max_length": 128,
        "padding": "max_length",
        "truncation": True,
        "add_special_tokens": True,
        "return_tensors": "pt"
    })

    # Optimizer / Criterion / Scheduler configuration
    optimizer_class: str = field(default=None)
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)

    criterion_class: str = field(default=None)
    criterion_kwargs: Dict[str, Any] = field(default_factory=dict)

    scheduler_class: str = field(default=None)
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)

    device: str = 'cuda'
    epochs: int = 5
    batch_size: int = 16


    # Dataset for the model
    data: str = field(default=None)

    # Logging
    save_dir: Path = Path("./experiments")
    experiment_name: str = "dummy"






    def __post_init__(self):
        """Validate required fields and ensure kwargs are dictionaries."""
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path is required in ExperimentConfig")
        if self.num_labels is None:
            raise ValueError("num_labels is required in ExperimentConfig")
        if self.strategy_class is None:
            raise ValueError("strategy_class is required in ExperimentConfig")
        if self.sampler_class is None:
            raise ValueError("sampler_class is required in ExperimentConfig")
        if self.optimizer_class is None:
            raise ValueError("optimizer_class is required in ExperimentConfig")
        if self.criterion_class is None:
            raise ValueError("criterion_class is required in ExperimentConfig")

        # Ensure kwargs are always dicts
        self.strategy_kwargs = self.strategy_kwargs or {}
        self.sampler_kwargs = self.sampler_kwargs or {}
        self.optimizer_kwargs = self.optimizer_kwargs or {}
        self.criterion_kwargs = self.criterion_kwargs or {}
        self.scheduler_kwargs = self.scheduler_kwargs or {}


# @dataclass
# class ExperimentConfig:
#     # Reproducibility
#     seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46]) # seeds for multiple runs to average results
#     # For our adaptive strategy
#     epsilons: List[float] = field(default_factory=lambda: [ 0.1, 0.25,0.5,0.75]) #minimum improvement in macro-F1
#     Ks: List[int] = field(default_factory=lambda: [ 2, 3, 6]) #how many consecutive rounds ΔF1 must stay below ϵ before we decide to switch
#     min_rounds_before_switch: int = 3
#     # Active learning
#     initial_pool_size: int = 200 # Initial size of the pool for training
#     acquisition_batch_size: int = 32 # new labels per round
#     # Training (per round)
#     batch_size: int = 16 # Batch size for model training
#     learning_rate: float = 2e-5 # Learning rate for model training
#     epochs: int = 5 # Number of epochs for model training
#     weight_decay: float = 0.01 # Weight decay for model training
#     optimizer: str = 'adamw' # Optimizer type or adam
#     # Strategy and Sampler instances
#     strategy: 'BaseStrategy' = field(default=None)
#     sampler: 'BaseSampler' = field(default=None)
#
#     # Model configuration (still need class + kwargs for model creation)
#     model_class: type = field(default=None)
#     model_kwargs: Dict[str, Any] = field(default_factory=dict)
#
#     # Datasets for the model
#     train_dataset: 'Dataset' = field(default=None)
#     val_dataset: 'Dataset' = field(default=None)
#     test_dataset: 'Dataset' = field(default=None)
#
#     # # Evaluation (dont think is needed)
#     # primary_metric: str = "macro_f1"
#
#     # Logging
#     save_dir: Path = Path("./experiments")
#     experiment_name: str = "dummy"
#
#     def __post_init__(self):
#         """Validate required fields."""
#         if self.strategy is None:
#             raise ValueError("strategy instance is required in config")
#         if self.sampler is None:
#             raise ValueError("sampler instance is required in config")
#         if self.model_class is None:
#             raise ValueError("model_class is required in config")