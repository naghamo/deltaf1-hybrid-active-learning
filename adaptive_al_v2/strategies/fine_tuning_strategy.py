import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple

from .base_strategy import BaseStrategy
from ..pool import DataPool


class FineTuneStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _train_implementation(self, model: Any, pool, new_indices: List[int]) -> Tuple[Any, Dict]:
        """
        Fine-tunes the model for epochs using all labeled data so far.
        """
        model.train()  # Set the model to training mode
        model.to(self.device)

        if new_indices:
            pool.add_labeled_samples(new_indices)

        # Get all labeled data from the pool
        labeled_subset = pool.get_labeled_subset()
        dataloader = DataLoader(labeled_subset, batch_size=self.batch_size, shuffle=True)

        # Training metrics
        total_loss = 0.0
        num_batches = 0

        # Training loop
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_batches = 0

            for batch in dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_batches += 1

            # Step scheduler if provided
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += epoch_loss
            num_batches += epoch_batches

        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Return model and training statistics
        training_stats = {
            "avg_loss": avg_loss,
            "epochs": self.epochs,
            "total_samples": len(labeled_subset),
            "new_samples": len(new_indices) if new_indices is not None else 0,
        }

        return model, training_stats