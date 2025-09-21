import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import List, Any

from tqdm import tqdm

from .base_sampler import BaseSampler
from ..pool import DataPool


class EntropySampler(BaseSampler):
    """Selects samples with the highest predictive entropy."""
    def __init__(self, show_progress=False, random_indices_fraction=0, **kwargs):
        super().__init__(**kwargs)
        self.show_progress = show_progress
        self.random_indices_fraction = random_indices_fraction


    def get_unlabeled_indices(self, pool:DataPool):
        return pool.get_unlabeled_indices()


    def select(self, pool: DataPool, num_samples: int) -> List[int]:
        self.model.eval()
        self.model.to(self.device)

        random_indices = self.get_random_unlabeled(pool, int(num_samples*self.random_indices_fraction))
        unlabeled_indices = self.get_unlabeled_indices(pool)
        unlabeled_indices = [i for i in unlabeled_indices if i not in random_indices]
        unlabeled_subset = pool.get_subset(unlabeled_indices)

        if not unlabeled_indices:
            return []

        dataloader = DataLoader(
            unlabeled_subset,
            batch_size=self.batch_size,
            shuffle=False
        )

        all_entropies = []
        # No need for all_indices list anymore

        iterator = dataloader
        if self.show_progress:
            iterator = tqdm(dataloader, desc="Entropy Sampling", leave=False)

        with torch.no_grad():
            for batch in iterator:
                inputs, _ = batch
                inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

                outputs = self.model(**inputs)
                logits = outputs['logits']

                probs = F.softmax(logits, dim=-1)

                log_probs = torch.log(probs)
                entropy_vals = -torch.sum(probs * log_probs, dim=1)

                all_entropies.append(entropy_vals.cpu())

        all_entropies = np.concatenate([t.numpy() for t in all_entropies])

        # Ensure we don't try to select more than we have
        num_to_select = min(num_samples - len(random_indices), len(all_entropies))
        top_k_indices = np.argpartition(all_entropies, num_to_select)[-num_to_select:]

        # Map these indices back to the original unlabeled_indices
        selected_pool_indices = [unlabeled_indices[i] for i in top_k_indices]

        selected_pool_indices.extend(random_indices)
        return selected_pool_indices

