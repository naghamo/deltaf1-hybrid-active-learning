from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .samplers import BaseSampler
    from .strategies import BaseStrategy
    from torch.utils.data import Dataset


@dataclass
class ExperimentConfig:
    # Reproducibility
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46]) # seeds for multiple runs to average results
    # For our adaptive strategy
    epsilons: List[float] = field(default_factory=lambda: [ 0.1, 0.25,0.5,0.75]) #minimum improvement in macro-F1
    Ks: List[int] = field(default_factory=lambda: [ 2, 3, 6]) #how many consecutive rounds ΔF1 must stay below ϵ before we decide to switch
    min_rounds_before_switch: int = 3
    # Active learning
    initial_pool_size: int = 200 # Initial size of the pool for training
    acquisition_batch_size: int = 32 # new labels per round
    # Training (per round)
    batch_size: int = 16 # Batch size for model training
    learning_rate: float = 2e-5 # Learning rate for model training
    epochs: int = 15 # Number of epochs for model training

    # Strategy and Sampler instances
    strategy: 'BaseStrategy' = field(default=None)
    sampler: 'BaseSampler' = field(default=None)

    # Model configuration (still need class + kwargs for model creation)
    model_class: type = field(default=None)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Datasets for the model
    train_dataset: 'Dataset' = field(default=None)
    val_dataset: 'Dataset' = field(default=None)
    test_dataset: 'Dataset' = field(default=None)

    # # Evaluation (dont think is needed)
    # primary_metric: str = "macro_f1"

    # Logging
    save_dir: Path = Path("./experiments")
    experiment_name: str = "dummy"

    def __post_init__(self):
        """Validate required fields."""
        if self.strategy is None:
            raise ValueError("strategy instance is required in config")
        if self.sampler is None:
            raise ValueError("sampler instance is required in config")
        if self.model_class is None:
            raise ValueError("model_class is required in config")