import torch
from torch.utils.data import Subset, DataLoader
from abc import ABC
from typing import List
from base_strategy import BaseStrategy


class FineTuneStrategy(BaseStrategy, ABC):
    def __init__(self, model, model_class, model_kwargs, pool, dataset, optimizer_class, optimizer_kwargs, criterion,
                 device='cpu', epochs=3, batch_size=16):
        super().__init__(model, model_class, model_kwargs, pool)
        self.dataset = dataset  # full dataset, supports indexing
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size

        self.model.to(self.device)

    def train_round(self, new_indices: List[int], round_i: int) -> None:
        """
        Fine-tunes the model for `x` epochs using only the new datapoints.
        """
        self.model.train()

        # Get only the new data
        subset = Subset(self.dataset, new_indices)
        dataloader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)

        # Create a new optimizer for fine-tuning
        optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)

        for epoch in range(self.epochs):
            for batch in dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
