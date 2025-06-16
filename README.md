# DeltaF1-Hybrid-Active-Learning

This project implements a hybrid training strategy for pool-based active learning. The strategy adaptively switches between retraining and fine-tuning a model based on the change in macro-F1 score (ΔF1) across rounds.

## 🌟 Goal

To reduce training cost in active learning without sacrificing performance, we propose a hybrid training strategy that switches from retraining to fine-tuning once ΔF1 falls below a small threshold ε for k consecutive rounds.

## 🔍 Datasets

We experiment on three text classification benchmarks:

- [AG News](https://huggingface.co/datasets/ag_news) – Topic classification (balanced)
- [IMDb Reviews](https://huggingface.co/datasets/imdb) – Sentiment analysis (binary)
- [Jigsaw Toxic Comments](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) – Multi-label toxicity classification (imbalanced)

## 🧠 Models

We use `DistilBERT` as the base encoder, fine-tuned separately per dataset with consistent hyperparameters.

## 🔁 Training Strategies

* `retrain`: Full retraining from scratch on all labeled data every round.
* `fine-tune`: Warm-start from the previous round and continue training on all labeled data.
* `new-only`: Fine-tune only on the newly labeled batch each round (no replay of past data).
* `hybrid`: Start with retraining, then switch to fine-tuning (on all labeled data) when ΔF1 < ε for $k$ consecutive rounds.

> All strategies use the same model architecture and hyperparameters to ensure fair comparison.

## 📊 Evaluation

- **Metric**: Macro-F1 score (primary)
- **Switch trigger**: ΔF1 < ε  for \( k \) rounds  
- **Other metrics**: training time, label efficiency  
- Results include plots of learning curves, ΔF1 trends, and switch timing.

## 🛠 Repo Structure
hybrid_active_learning/
├── data/                 # Raw and preprocessed datasets (AG News, IMDb, Jigsaw)
├── media/
├── scripts/              # All source code
│   ├── utils.py              # Shared helpers: seeding, timing, plotting, etc.
│   ├── 
│   └── 
├── models/              # saved model checkpoints
├── results/             # Metrics logs, ΔF1 values, plots
├── README.md            # Project description and setup guide
├── requirements.txt     # List of dependencies
├── eda_preprocessing/
│   ├── dataset_eda.ipynb  # Initial exploration of AG News, IMDb, Jigsaw
│   ├── split_saver.ipynb # Creating and saving train/val/test splits
│   ├── 
│   └── 
└── config.json          # Experiment config: ε, k, batch size, etc.



## ⚙️ Requirements

- Python 3.10
- PyTorch
- scikit-learn
- datasets
- matplotlib, pandas, tqdm, numpy
- 

To install dependencies:

```bash
pip install -r requirements.txt
