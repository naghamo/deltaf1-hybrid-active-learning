"""
Model evaluation utilities for active learning experiments.

This module provides functions for evaluating PyTorch models, including
full dataset evaluation, approximate subset-based evaluation for efficiency,
variance analysis across multiple evaluations, and confusion matrix computation.
"""

import logging
from typing import Optional, Dict, List, Tuple
import random

import torch
import numpy as np
import time
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Subset


def _evaluate_model_core(
        model: torch.nn.Module,
        criterion,
        batch_size: int,
        dataset: torch.utils.data.Dataset,
        device: str = "cuda",
        subset_size: Optional[int] = None,
        random_seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Core evaluation function that handles both full and subset evaluation.

    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion: Loss function.
        batch_size (int): Batch size for evaluation.
        dataset (torch.utils.data.Dataset): Dataset to evaluate on.
        device (str): Device for evaluation.
        subset_size (Optional[int]): If provided, evaluate on random subset of this size.
        random_seed (Optional[int]): Seed for reproducible subset selection.

    Returns:
        Dict[str, float]: Evaluation metrics.
    """
    start = time.perf_counter()

    # Create subset if requested
    eval_dataset = dataset
    if subset_size is not None:
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)

        indices = random.sample(range(len(dataset)), min(subset_size, len(dataset)))
        eval_dataset = Subset(dataset, indices)

    loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
            targets = targets.to(device)

            outputs = model(**inputs)
            logits = outputs['logits']
            loss = criterion(logits, targets)
            total_loss += loss.item()

            # Expecting multi-class (not binary)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(targets.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    metrics = {
        "loss": total_loss / len(loader),
        "f1_score": f1_score(all_labels, all_preds, average="macro"),
        "accuracy": accuracy_score(all_labels, all_preds)
    }

    subset_info = f" on subset of {len(eval_dataset)} samples" if subset_size else ""
    logging.info(f"Model evaluation{subset_info} took {time.perf_counter() - start:.2f} seconds")
    return metrics


def evaluate_model(
        model: torch.nn.Module,
        criterion,
        batch_size: int,
        dataset: Optional[torch.utils.data.Dataset] = None,
        device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate a PyTorch model on a given dataset.

    This function computes the average loss, macro F1-score, and accuracy
    of the model on the provided dataset.

    Args:
        model (torch.nn.Module):
            The trained (or partially trained) model to evaluate.
        criterion (torch.nn.Module or callable):
            Loss function used to compute evaluation loss (e.g., CrossEntropyLoss).
        batch_size (int):
            Batch size to use when evaluating.
        dataset (torch.utils.data.Dataset, optional):
            Dataset to evaluate the model on. Must return (inputs, targets) where
            inputs is a dictionary of tensors and targets is a tensor of labels.
        device: Device on which to perform evaluation.

    Returns:
        Dict[str, float]: A dictionary containing:
            - "loss": Average loss over the dataset.
            - "f1_score": Macro-averaged F1 score across all classes.
            - "accuracy": Overall classification accuracy.
    """
    if dataset is None:
        raise ValueError("Dataset must be provided for evaluation")

    return _evaluate_model_core(model, criterion, batch_size, dataset, device)


def approximate_evaluate_model(
        model: torch.nn.Module,
        criterion,
        batch_size: int,
        dataset: torch.utils.data.Dataset,
        subset_size: int,
        device: str = "cuda",
        random_seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate a PyTorch model on a random subset of the dataset for faster evaluation.

    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion: Loss function.
        batch_size (int): Batch size for evaluation.
        dataset (torch.utils.data.Dataset): Full dataset to sample from.
        subset_size (int): Size of random subset to evaluate on.
        device (str): Device for evaluation.
        random_seed (Optional[int]): Seed for reproducible results.

    Returns:
        Dict[str, float]: Evaluation metrics on the subset.
    """
    return _evaluate_model_core(
        model, criterion, batch_size, dataset, device,
        subset_size=subset_size, random_seed=random_seed
    )


def approximate_evaluate_variance(
        model: torch.nn.Module,
        criterion,
        batch_size: int,
        dataset: torch.utils.data.Dataset,
        subset_size: int,
        num_evaluations: int = 5,
        device: str = "cuda",
        base_seed: Optional[int] = None
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Perform multiple approximate evaluations and compute variance in metrics.

    This function runs approximate evaluation multiple times with different random
    subsets and returns both individual results and variance statistics.

    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion: Loss function.
        batch_size (int): Batch size for evaluation.
        dataset (torch.utils.data.Dataset): Full dataset to sample from.
        subset_size (int): Size of random subset for each evaluation.
        num_evaluations (int): Number of evaluations to perform.
        device (str): Device for evaluation.
        base_seed (Optional[int]): Base seed for reproducible results.

    Returns:
        Tuple[List[Dict[str, float]], Dict[str, float]]:
            - List of individual evaluation results
            - Dictionary with mean and std for each metric
    """
    results = []

    for i in range(num_evaluations):
        seed = base_seed + i if base_seed is not None else None
        result = approximate_evaluate_model(
            model, criterion, batch_size, dataset, subset_size, device, seed
        )
        results.append(result)

    # Compute variance statistics
    metrics_arrays = {}
    for metric in results[0].keys():
        metrics_arrays[metric] = np.array([result[metric] for result in results])

    variance_stats = {}
    for metric, values in metrics_arrays.items():
        variance_stats[f"{metric}_mean"] = float(np.mean(values))
        variance_stats[f"{metric}_std"] = float(np.std(values))
        variance_stats[f"{metric}_var"] = float(np.var(values))

    logging.info(f"Completed {num_evaluations} approximate evaluations with subset size {subset_size}")
    logging.info("Variance statistics:")
    for metric in ['loss', 'f1_score', 'accuracy']:
        mean_val = variance_stats[f"{metric}_mean"]
        std_val = variance_stats[f"{metric}_std"]
        logging.info(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")

    return results, variance_stats



def compute_confusion_matrix(
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        device: str = "cuda",
        normalize: bool = False,
        class_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Compute and optionally plot the confusion matrix for a model.

    Args:
        model (torch.nn.Module): Trained model.
        dataset (torch.utils.data.Dataset): Dataset to evaluate on.
        batch_size (int): Batch size for DataLoader.
        device (str): Device for evaluation.
        normalize (bool): Whether to normalize counts per class.
        class_names (Optional[List[str]]): List of class names for plotting.

    Returns:
        np.ndarray: Confusion matrix (optionally normalized).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
            targets = targets.to(device)

            outputs = model(**inputs)
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(targets.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    cm = confusion_matrix(all_labels, all_preds, normalize="true" if normalize else None).tolist()

    return cm