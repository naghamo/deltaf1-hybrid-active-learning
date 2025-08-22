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
    seed: int = 42

    # Active learning
    initial_pool_size: int = 100 # Initial size of the pool for training
    batch_size: int = 50 # Batch of new data for sampler to retreive

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