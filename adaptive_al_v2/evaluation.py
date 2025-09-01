from typing import Optional, Dict

import torch

from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader

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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    all_preds, all_labels = [], []

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

    return metrics