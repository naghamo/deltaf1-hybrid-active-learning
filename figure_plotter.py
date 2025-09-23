import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

dataset_names = {"agnews": "AG News", "imdb": "IMDb", "jigsaw": "Jigsaw"}
strategy_names = {"NewOnlyStrategy": "New only", "RetrainStrategy": "Retrain", "FineTuneStrategy": "Fine tuning",
                  "DeltaF1Strategy": "HybridAL"}

def plot_f1_vs_time(results_paths: list, save_path: str, cmap:str='tab20b'):
    """
    Plots test set F1 score vs training time for different experiments.

    Parameters:
    - results_paths: List of file paths containing results for each experiment in .json format.
    - save_path: Path to save the generated plot.
    - cmap: Colormap to use for different lines in the plot.
    """

    plt.figure(figsize=(6, 4))

    for i, path in enumerate(results_paths):
        with open(path, 'r') as f:
            data = json.load(f)
        strat = data['cfg']['strategy_class']
        dataset = data['cfg']['data']

        round_val_stats = data['round_val_stats']

        overall_time = 0
        times, f1_scores = [], []
        for round in round_val_stats:
            overall_time += round['training_time']
            times.append(overall_time)
            f1_scores.append(round['f1_score'])

        label = f"{dataset_names[dataset]}/{strategy_names[strat]}"
        color = plt.get_cmap(cmap)(i / 20)

        plt.plot(times, f1_scores, marker='o', label=label, color=color)

    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Test Set Macro-F1 Score')
    plt.title('Test Set Macro-F1 Score vs Training Time')
    plt.legend()
    plt.savefig(save_path)
    plt.close()