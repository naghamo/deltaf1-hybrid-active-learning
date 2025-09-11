
# HybridAL: Adaptive Training Strategy Switching in Pool-Based Active Learning Using ΔF1  

## Overview  
This repository contains the code and experiments for **HybridAL**, an adaptive active learning framework for text classification.  
Instead of training with a fixed strategy (retrain or fine-tune) in all active learning rounds, HybridAL dynamically **switches strategies** based on validation performance improvements (ΔF1).  

- **Retrain**: robust, prevents bias but costly.  
- **Fine-tune**: efficient, but risks drift/overfitting.  
- **HybridAL (proposed)**: start with retrain, switch to fine-tune when ΔF1 stabilizes.  

This approach reduces compute cost while maintaining accuracy.  
We evaluate HybridAL on **AG News**, **IMDb Reviews**, and **Jigsaw Toxic Comments**.  

---
## Pipeline  

The project follows the standard **pool-based active learning pipeline**:  

1. **Initialization** – Start with a small labeled dataset and a large pool of unlabeled samples.  
2. **Model Training** – Train a model on the current labeled pool.  
3. **Evaluation** – Evaluate the model on a validation set to monitor performance.  
4. **Querying / Sampling** – Select the most informative samples from the unlabeled pool (e.g., random, entropy, uncertainty).  
5. **Annotation** – Add the newly labeled samples to the labeled pool.  
6. **Iteration** – Repeat training, evaluation, and querying until the budget (rounds or labels) is exhausted.  
7. **Final Testing** – Evaluate the final model on the held-out test set.  

HybridAL extends this pipeline by **adaptively switching the training strategy** (from retraining to fine-tuning) when performance improvements (ΔF1) stabilize, reducing computational costs while preserving accuracy.

![Pipeline](media/active_learning_pipeline.png)  

---

## Repository Structure  
```

adaptive_al/
│
├── strategies/          # Training strategies (retrain, fine-tune, new-only, deltaf1)
│   ├── base\_strategy.py
│   ├── retrain\_strategy.py
│   ├── fine\_tuning\_strategy.py
│   ├── new\_only\_strategy.py
│   ├── deltaf1\_strategy.py
│
├── samplers/            # Active learning samplers (uncertainty, entropy, etc.)
│
├── utils/               # Pipeline utilities
│   ├── active\_learning.py
│   ├── config.py
│   ├── evaluation.py
│   ├── pool.py
│
├── data/                # Datasets (AG News, IMDb, Jigsaw)
│
├── eda\_preprocessing/   # EDA and preprocessing notebooks
│   ├── 01\_agnews\_eda.ipynb
│   ├── 02\_imdb\_eda.ipynb
│   ├── 03\_jigsaw\_eda.ipynb
│
├── experiments/         # Experiment scripts and results
├── media/               # Figures, plots, diagrams
├── run\_example.ipynb    # Example run of the pipeline
├── requirements.txt     # Python dependencies
└── README.md

````

---

## Setup Instructions  

### Requirements  
- Python 3.11  
- GPU recommended (tested on NVIDIA Tesla M60 in Azure VM)  
- Dependencies are listed in `requirements.txt`  

### Installation  
```bash
git clone https://github.com/naghamo/deltaf1-hybrid-active-learning.git
cd deltaf1-hybrid-active-learning
pip install -r requirements.txt
````

---

## How to Run

The full active learning pipeline is implemented in `adaptive_al_v2/active_learning.py` using two main classes:  
- **ExperimentConfig** – defines the experiment settings (dataset, strategy, model, training parameters).  
- **ActiveLearning** – manages the active learning loop (train, sample, evaluate, save results).  

### Example Usage  

```python
import torch
from pathlib import Path
from adaptive_al_v2.active_learning import ActiveLearning, ExperimentConfig

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define experiment configuration
cfg = ExperimentConfig(
    seed=42,
    total_rounds=5,
    experiment_name="dummy_test_pipeline",
    save_dir=Path("./experiments"),

    # Pool settings
    initial_pool_size=200,
    acquisition_batch_size=256,

    # Model
    model_name_or_path="distilbert-base-uncased",
    num_labels=4,
    tokenizer_kwargs={
        "max_length": 128,
        "padding": "max_length",
        "truncation": True,
        "add_special_tokens": True,
        "return_tensors": "pt"
    },

    # Dataset
    data="agnews",

    # Strategy
    strategy_class="DeltaF1Strategy",
    strategy_kwargs={"epsilon": 0.01, "k": 2},

    # Optimizer / Loss / Scheduler
    optimizer_class="Adam",
    optimizer_kwargs={"lr": 1e-3, "weight_decay": 1e-4},
    criterion_class="CrossEntropyLoss",
    criterion_kwargs={},
    scheduler_class="StepLR",
    scheduler_kwargs={"step_size": 10, "gamma": 0.1},

    # Sampler
    sampler_class="RandomSampler",
    sampler_kwargs={"seed": 42},

    # Training
    device=device,
    epochs=3,
    batch_size=64
)

# Initialize active learning pipeline
al = ActiveLearning(cfg)

# Run full pipeline
final_metrics = al.run_full_pipeline()
print(f"Final Test Metrics: F1={final_metrics['f1_score']:.4f}, "
      f"Accuracy={final_metrics['accuracy']:.4f}, "
      f"Loss={final_metrics['loss']:.4f}")

# Save experiment results
al.save_experiment()
````

After running, results (metrics, config, logs) are saved under `./experiments/<experiment_name>/`.




---

## Datasets

| Dataset | Task Type                    | Size (Train/Test) | #Classes | Notes      |
| ------- | ---------------------------- |-------------------|----------| ---------- |
| AG News | Topic classification         | 120k / 7.6k       | 4        | Balanced   |
| IMDb    | Sentiment analysis           | 25k               | 2        | Balanced   |
| Jigsaw  | Toxic comment classification | 160k              | 6        | Imbalanced |

Preprocessing and label distributions are analyzed in the `eda_preprocessing/` notebooks.

---

## Reproducibility

* No values are hardcoded in training scripts.
* Experiments are run with **five random seeds {42, 43, 44, 45, 46}**, and results are averaged.
* Results, figures, and logs are saved under `experiments/` and `media/`.


---

## Team

* **Nagham Omar**
* **Evgeny Mishliyakov**
* **Vsevolod Rusanov**
* **Maya Rozenshtein**

Technion – Israel Institute of Technology

---

## Citation

If you use this work, please cite:

```
@misc{omar2025hybridal,
  title={HybridAL: Adaptive Training Strategy Switching in Pool-Based Active Learning Using ΔF1},
  author={Nagham Omar and Evgeny Mishliyakov and Vsevolod Rusanov and Maya Rozenshtein},
  year={2025},
  institution={Technion – Israel Institute of Technology},
  howpublished={\url{https://github.com/naghamo/deltaf1-hybrid-active-learning}}
}
```

---

✨ This repository contains all source code, experiments, and results for HybridAL.

```

