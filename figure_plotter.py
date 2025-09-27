import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

dataset_names = {"agnews": "AG News", "imdb": "IMDb", "jigsaw": "Jigsaw"}
strategy_names = {"NewOnlyStrategy": "New only", "RetrainStrategy": "Retrain", "FineTuneStrategy": "Fine tuning",
                  "DeltaF1Strategy": "HybridAL"}


def filter_experiments_df(df: pd.DataFrame, dataset: str, strategy: str, **filter_kwargs):
    """
    Filters the experiments DataFrame for a specific dataset and strategy. If the strategy is 'DeltaF1Strategy',
    it also filters based on provided hyperparameters.

    Parameters:
    - df: The DataFrame containing experiment results.
    - dataset: The dataset to filter by ('agnews', 'imdb', 'jigsaw').
    - strategy: The strategy to filter by ('NewOnlyStrategy', 'RetrainStrategy', 'FineTuneStrategy', 'DeltaF1Strategy').
    - filter_kwargs: Additional keyword arguments for filtering, such as 'seed' or hyperparameters for 'DeltaF1Strategy'.

    Returns:
    - A filtered pandas DataFrame.
    """
    mask = (df['dataset'] == dataset) & (df['strategy'] == strategy)
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
        dataset: str,
        save_path: str = None,
        cmap: str = "tab20b",
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
    - dataset: The dataset to plot its strategies results ('agnews', 'imdb', 'jigsaw').
    - save_path: Optional path to save the plot image. If None, the plot is just shown.
    - cmap: Colormap name for strategies.
    - n_grid: Number of points in the common time grid for interpolation.
    - show_individual: Whether to plot faint individual seed F1 curves in the background.
    """

    plt.figure(figsize=(6, 4))
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
                epsilon=hybrid_hyper.get['epsilon'] if is_delta_f1 else None,
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

        if len(seed_curves) == 0:
            continue

        # Set the common time grid to be between 0 and the minimum max time across seeds
        max_times = [curve[0][-1] for curve in seed_curves]
        t_max_common = np.min(max_times)
        if t_max_common <= 0:
            continue

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
    plt.ylabel('Test Set Macro-F1 Score')
    plt.title(f'Test Set Macro-F1 vs Training Time (mean across seeds) for {dataset_names[dataset]}')
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()
