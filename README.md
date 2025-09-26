
# HybridAL: Adaptive Training Strategy Switching in Pool-Based Active Learning Using ΔF1  

## Overview  
This repository contains the code and experiments for **HybridAL**, an adaptive active learning framework for text classification.  
Instead of training with a fixed strategy (retrain or fine-tune) in all active learning rounds, HybridAL dynamically **switches strategies** based on validation performance improvements (ΔF1).  
- **New-only**: forgets past knowledge
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

## Repository Structure  (fix this section**************&&&&&&&&&&&***********)
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
- GPU recommended
- Dependencies are listed in `requirements.txt`  

### Installation  
```bash
git clone https://github.com/naghamo/deltaf1-hybrid-active-learning.git
cd deltaf1-hybrid-active-learning
pip install -r requirements.txt
````

---

## How to Run
# TODO:*****************************************************************

### Example Usage


````
# TODO:*****************************************************
After running, results (metrics, config, logs) are saved under `./experiments/<experiment_name>/`.




---

## Datasets

| Dataset | Task Type                    | Size (Train/Test) | #Classes        | Notes      |
| ------- | ---------------------------- |-------------------|-----------------| ---------- |
| AG News | Topic classification         | 120k / 7.6k       | 4               | Balanced   |
| IMDb    | Sentiment analysis           | 25k               | 2               | Balanced   |
| Jigsaw  | Toxic comment classification | 160k              | 2               | Imbalanced |

Preprocessing and label distributions are analyzed in the `eda_preprocessing/` notebooks.

---
# TODO: *******************************************************************************************
## Reproducibility

* No values are hardcoded in training scripts.
* Experiments are run with **five random seeds {42, 43, 44, 45, 46}**, and results are averaged.


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

✨ This repository contains all source code.


