import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

sns.set_theme(style="whitegrid")

dataset_names = {"agnews": "AG News", "imdb": "IMDb", "jigsaw": "Jigsaw"}
dataset_labels = {"agnews": ["World", "Sports", "Business", "Sci/Tech"],
                  "imdb": ["Negative", "Positive"],
                  "jigsaw": ["Clean", "Toxic"]}
switch_rounds = {"agnews": {42: 10, 43: 10, 44: 10}, "imdb": {42: 12, 43: 9, 44: 11}, "jigsaw": {42: 19, 43: 11,
                                                                                                 44: 6}}  # The rounds on which the switch was made in the best hybrid configuration
strategy_names = {"NewOnlyStrategy": "New only", "RetrainStrategy": "Retrain", "FineTuneStrategy": "Fine tuning",
                  "DeltaF1Strategy": "HybridAL"}
dpi = 250  # DPI for saving figures
figsize = (6, 4)  # Default figure size


def filter_experiments_df(df: pd.DataFrame, **filter_kwargs):
    """
    Filters the experiments DataFrame for a specific dataset and strategy. If the strategy is 'DeltaF1Strategy',
    it also filters based on provided hyperparameters.

    Parameters:
    - df: The DataFrame containing experiment results.
    - filter_kwargs: keyword arguments for filtering, such as 'dataset', 'strategy', 'seed' or hyperparameters for 'DeltaF1Strategy'.

    Returns:
    - A filtered pandas DataFrame.
    """
    mask = pd.Series(True, index=df.index)
    if filter_kwargs:
        for key, value in filter_kwargs.items():
            if key in df.columns and value:
                mask &= (df[key] == value)

    return df.loc[mask]


def get_experiments_df(main_results_path: str):
    """
    Reads all experiment result files from the specified directory and compiles them into a single DataFrame.

    Parameters:
    - main_results_path: Path to the directory containing experiment result files in .json format.

    Returns:
    - A pandas DataFrame containing all experiments' data.
    """
    all_data = []
    for root, dirs, files in os.walk(main_results_path):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                    data['folder_name'] = os.path.basename(root)
                all_data.append(data)
                break

    df = pd.json_normalize(all_data, sep='_')

    cols_to_keep = ['total_rounds', 'round_val_stats', 'cfg_seed', 'cfg_strategy_class', 'cfg_strategy_kwargs_epsilon',
                    'cfg_strategy_kwargs_k', 'cfg_strategy_kwargs_validation_fraction', 'cfg_data',
                    'final_pool_stats_labeled_count',
                    'final_pool_stats_unlabeled_count', 'final_test_stats_loss', 'final_test_stats_f1_score',
                    'final_test_stats_accuracy', 'confusion_matrix', 'folder_name']
    df = df[cols_to_keep]
    df.rename(columns={
        'cfg_seed': 'seed',
        'cfg_strategy_class': 'strategy',
        'cfg_strategy_kwargs_epsilon': 'epsilon',
        'cfg_strategy_kwargs_k': 'k',
        'cfg_strategy_kwargs_validation_fraction': 'validation_fraction',
        'cfg_data': 'dataset',
        'final_test_stats_f1_score': 'test_f1_score',
        'final_test_stats_accuracy': 'test_accuracy',
        'final_test_stats_loss': 'test_loss',
        'final_pool_stats_labeled_count': 'labeled_count',
        'final_pool_stats_unlabeled_count': 'unlabeled_count',
    }, inplace=True)
    return df


def get_summary_table(experiments_df: pd.DataFrame, hybrid_hyper: dict):
    summary_data = []

    for dataset in experiments_df['dataset'].unique():
        for strategy in experiments_df['strategy'].unique():
            is_delta_f1 = (strategy == 'DeltaF1Strategy')

            filtered_df = filter_experiments_df(experiments_df, dataset=dataset, strategy=strategy,
                                                epsilon=hybrid_hyper['epsilon'] if is_delta_f1 else None,
                                                k=hybrid_hyper['k'] if is_delta_f1 else None,
                                                validation_fraction=hybrid_hyper[
                                                    'validation_fraction'] if is_delta_f1 else None)

            training_times = []
            for _, row in filtered_df.iterrows():
                round_val_stats = row['round_val_stats']
                total_time = sum(round['training_time'] for round in round_val_stats)
                training_times.append(total_time)

            avg_f1 = filtered_df['test_f1_score'].mean()
            avg_time = np.mean(training_times)
            avg_total_rounds = filtered_df['total_rounds'].mean()

            summary_data.append({
                'Dataset': dataset_names.get(dataset, dataset),
                'Strategy': strategy_names.get(strategy, strategy),
                'Avg Test Set Macro-F1 Score': f"{avg_f1:.4f}",
                'Avg Training Time (sec)': f"{avg_time:.2f}",
                'Avg Total Rounds': f"{avg_total_rounds:.2f}"
            })

    summary_df = pd.DataFrame(summary_data)
    return summary_df


def plot_f1_vs_time_avg(
        experiments_df: pd.DataFrame,
        hybrid_hyper: dict,
        save_dir_path: str = None,
        cmap: str = "Set2",
        n_grid: int = 250,
        show_individual: bool = False
):
    """
    Plots Macro-F1 vs cumulative training time, averaged across seeds per strategy, with interpolation onto a common
    time grid and a variability band.

    Parameters:
    - experiments_df: DataFrame containing experiment results, in the format of get_experiments_df().
    - hybrid_hyper: Dictionary with hyperparameters for the best DeltaF1Strategy
    (keys: 'epsilon', 'k', 'validation_fraction').
    - save_dir_path: Optional directory path to save the plot images. If None, the plots are just shown.
    - cmap: Colormap name for strategies.
    - n_grid: Number of points in the common time grid for interpolation.
    - show_individual: Whether to plot faint individual seed F1 curves in the background.
    """

    for dataset in experiments_df['dataset'].unique():
        plt.figure(figsize=figsize)
        color_map = plt.get_cmap(cmap)
        strategies = list(strategy_names.keys())

        for s_i, strategy in enumerate(strategies):
            # Gather the time points and F1 scores for each seed
            seed_curves = []
            for seed in sorted(experiments_df['seed'].unique()):
                is_delta_f1 = (strategy == 'DeltaF1Strategy')

                row = filter_experiments_df(
                    experiments_df, dataset=dataset,
                    strategy=strategy,
                    seed=seed,
                    epsilon=hybrid_hyper['epsilon'] if is_delta_f1 else None,
                    k=hybrid_hyper['k'] if is_delta_f1 else None,
                    validation_fraction=hybrid_hyper['validation_fraction'] if is_delta_f1 else None
                )

                # Build cumulative time curve
                round_val_stats = row['round_val_stats'].values[0]
                times, f1s = [], []
                t = 0.0
                for r in round_val_stats:
                    t += float(r['training_time'])
                    times.append(t)
                    f1s.append(float(r['f1_score']))

                if len(times) >= 2:
                    seed_curves.append((times, f1s))

            # Set the common time grid to be between 0 and the minimum max time across seeds
            max_times = [curve[0][-1] for curve in seed_curves]
            t_max_common = np.min(max_times)

            grid = np.linspace(0.0, t_max_common, n_grid)

            # Interpolate each seed's F1 scores onto the common time grid
            interp_f1s = []
            for times, f1s in seed_curves:
                f_on_grid = np.interp(grid, times, f1s)
                interp_f1s.append(f_on_grid)

                if show_individual:
                    plt.plot(times, f1s, color=color_map(s_i / len(color_map.colors)), alpha=0.25, lw=1)

            interp_f1s = np.vstack(interp_f1s)
            mean_f1 = interp_f1s.mean(axis=0)

            # Calculate standard deviation for the band
            std = interp_f1s.std(axis=0)
            lower, upper = mean_f1 - std, mean_f1 + std

            # Plot mean and band per strategy
            label = strategy_names[strategy]
            color = color_map(s_i / len(color_map.colors))
            plt.plot(grid, mean_f1, label=label, color=color, lw=2.0)
            plt.fill_between(grid, lower, upper, color=color, alpha=0.15, linewidth=0)

        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Validation Set Macro-F1 Score')
        plt.title(f'Validation Set Macro-F1 vs Training Time (mean across seeds) for {dataset_names[dataset]}')
        plt.grid(True, alpha=0.25)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        if save_dir_path:
            plt.savefig(os.path.join(save_dir_path, f'f1_vs_time_{dataset}.png'), dpi=dpi)
        plt.show()


def plot_hybrid_hyper_variations(experiments_df: pd.DataFrame, best_hybrid_hyper: dict, save_dir_path: str = None,
                                 cmap: str = 'tab10'):
    """
    Plots 3 figures, where in each figure two of the three hybridAL hyperparameters are fixed to the best values,   and the third is varied.
    The figures show the mean test set F1 score vs the varied hyperparameter, with one line per dataset.

    Parameters:
    - experiments_df: DataFrame containing the experimental results.
    - best_hybrid_hyper: Dictionary containing the best hyperparameters for hybridAL.
    - save_dir_path: DIRECTORY path to save the generated plots. If None, plots are just shown.
    """
    hyperparams_names = {"epsilon": r"$\varepsilon$",
                         "k": r"$k$",
                         "validation_fraction": r"$c$"}

    for param in best_hybrid_hyper.keys():
        plt.figure(figsize=figsize)
        color_map = plt.get_cmap(cmap)

        for i, dataset in enumerate(experiments_df['dataset'].unique()):
            subset = experiments_df[
                (experiments_df['strategy'] == 'DeltaF1Strategy') &
                (experiments_df['dataset'] == dataset)
                ]

            for other_param, value in best_hybrid_hyper.items():
                if other_param != param:
                    subset = subset[subset[other_param] == value]

            means = subset.groupby(param)['test_f1_score'].mean()
            plt.plot(means.index, means.values, marker='o', label=dataset_names.get(dataset, dataset),
                     color=color_map(i))

        plt.xlabel(hyperparams_names[param])
        plt.ylabel('Mean Test Set Macro-F1 Score')

        title_postfix = ', '.join([f"{hyperparams_names[k]}={v}" for k, v in best_hybrid_hyper.items() if k != param])
        plt.title(f'Mean Test Set Macro-F1 Score vs {hyperparams_names[param]} ({title_postfix})')

        plt.legend(frameon=False)
        plt.grid(alpha=0.25)
        plt.tight_layout()

        if save_dir_path:
            plt.savefig(os.path.join(save_dir_path, f'hybrid_hyper_variation_{param}.png'), dpi=dpi)

        plt.show()


def plot_test_f1_bar_chart(experiments_df: pd.DataFrame, best_hybrid_hyper: dict, save_path: str = None):
    """
    Plots a bar chart comparing the mean test set macro-F1 scores of different strategies, averaged across all datasets and seeds.

    Parameters:
    - experiments_df: DataFrame containing experiment results.
    - best_hybrid_hyper: Dictionary containing the best hyperparameters for the HybridAL strategy
    - save_path: Optional path to save the generated plot.
    """
    results = []

    plt.figure(figsize=figsize)

    # Fixed color mapping to match the f1 vs time plot
    cmap = plt.get_cmap("Set2")
    strategy_colors = {
        "New only": cmap(0),
        "Retrain": cmap(1),
        "Fine tuning": cmap(2),
        "HybridAL": cmap(3),
    }

    for strategy in experiments_df['strategy'].unique():
        info = filter_experiments_df(experiments_df, strategy=strategy,
                                     **best_hybrid_hyper) if strategy == 'DeltaF1Strategy' else filter_experiments_df(
            experiments_df, strategy=strategy)

        mean_f1 = info['test_f1_score'].mean()
        std_f1 = info['test_f1_score'].std()
        name = strategy_names[strategy]

        results.append((name, mean_f1, std_f1))

    # Sort by mean F1 score
    results.sort(key=lambda x: x[1])  # ascending order

    bars = []
    # Plot Sorted bars
    for name, mean_f1, std_f1 in results:
        color = strategy_colors[name]
        bar = plt.bar(name, mean_f1, yerr=std_f1, capsize=5, color=color)
        bars.append((bar, mean_f1, std_f1))

    # Print bar values atop bars
    for bar, mean, std in bars:
        for rect in bar:
            plt.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + std + 0.02,
                f"{mean:.3f}",
                ha="center", va="bottom", fontsize=9
            )

    plt.ylabel('Mean Test Set Macro-F1 Score')
    plt.title('Comparison of Strategies on Mean Test Set Macro-F1 Score')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    plt.show()


def plot_confusion_heatmaps_hybrid(experiments_df: pd.DataFrame, hybrid_hyper: dict, save_dir_path: str = None):
    """
    Plots 3 average normalized confusion matrix heatmaps for the hybrid strategy, one per dataset.

    Parameters:
    - experiments_df: DataFrame containing experiments results.
    - hybrid_hyper: Dictionary containing the hyperparameters for the hybrid strategy.
    - save_dir_path: Directory path to save the generated plots.
    """

    for dataset in experiments_df['dataset'].unique():
        relevant_hybrid = filter_experiments_df(
            experiments_df,
            strategy='DeltaF1Strategy',
            dataset=dataset,
            epsilon=hybrid_hyper['epsilon'],
            k=hybrid_hyper['k'],
            validation_fraction=hybrid_hyper['validation_fraction']
        )

        # Sum the confusion matrices
        num_labels = len(relevant_hybrid['confusion_matrix'].values[0])
        sum_cm = np.zeros((num_labels, num_labels))
        for cm in relevant_hybrid['confusion_matrix']:
            sum_cm += np.array(cm)

        # Normalize the summed confusion matrix
        avg_cm = sum_cm / sum_cm.sum(axis=1, keepdims=True)

        # Round and renormalize again to avoid floating point issues
        avg_cm = np.round(avg_cm, 2)
        avg_cm = avg_cm / avg_cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            avg_cm, annot=True, fmt=".2f", cmap="Blues",
            cbar=True, ax=ax,
            cbar_kws={"shrink": 0.8, "label": "Proportion"}
        )

        # Rename the ticks to be the label meanings
        ax.set_xticklabels(dataset_labels[dataset], rotation=45, ha='right')
        ax.set_yticklabels(dataset_labels[dataset], rotation=0)

        plt.title(f'Normalized Global Confusion Matrix (HybridAL) - {dataset_names[dataset]}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        if save_dir_path:
            plt.savefig(os.path.join(save_dir_path, f'cm_hybrid_{dataset}.png'), dpi=dpi)
        plt.show()


def plot_f1_vs_round_switch(experiments_df: pd.DataFrame, best_hybrid_hyper: dict, save_dir_path: str = None):
    """
    Plots 3 plots of test set F1 score vs round number for HybridAL. There's a plot per dataset, and in each plot there is a line per seed. The round that the switch happened is marked.

    Parameters:
    - experiments_df: DataFrame containing the experiments' data.
    - best_hybrid_hyper: Dictionary containing the best hyperparameters for HybridAL.
    - save_dir_path: Directory path to save the generated plots. If None, plots are just shown.
    """

    for dataset in experiments_df['dataset'].unique():
        plt.figure(figsize=figsize)
        cmap = ListedColormap(['#9195F6', '#FB88B4', '#79d955'])

        color_i = 0
        total_rounds = []
        for j in experiments_df['seed'].unique():
            info = filter_experiments_df(
                experiments_df,
                dataset=dataset,
                strategy='DeltaF1Strategy',
                epsilon=best_hybrid_hyper['epsilon'],
                k=best_hybrid_hyper['k'],
                validation_fraction=best_hybrid_hyper['validation_fraction'],
                seed=j
            )

            f1_scores = [round['f1_score'] for round in info['round_val_stats'].values[0]]

            label = f"Seed {j}"
            color = cmap.colors[color_i]
            color_i += 1
            switch_round = switch_rounds[dataset][j]
            total_rounds.append(info['total_rounds'].values[0])

            plt.plot(range(1, len(f1_scores) + 1), f1_scores, label=label, color=color)
            plt.scatter(switch_round, f1_scores[switch_round - 1], color=color, s=50, edgecolor='black', zorder=5,
                        label=f"Switch Round (Seed {j})")

        plt.xlabel('Round Number')
        plt.ylabel('Validation Set Macro-F1 Score')
        plt.xticks(range(1, max(total_rounds) + 1), fontsize=10)
        plt.title(f'Validation Set Macro-F1 Score vs Round Number (HybridAL) - {dataset_names[
            dataset]}')
        plt.legend(fontsize='x-small')
        plt.grid(alpha=0.4)
        if save_dir_path is not None:
            plt.savefig(os.path.join(save_dir_path, f'f1_vs_round_hybrid_{dataset}.png'), dpi=dpi)
        plt.show()
