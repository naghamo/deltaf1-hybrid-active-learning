from typing import Any

import torch
from torch.utils.data import Dataset

class TextClassificationDataset(Dataset):
    """
    A custom dataset class for text classification.
    """
    def __init__(self, texts: list[str], labels: list[int], tokenizer: Any, tokenizer_kwargs: dict):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text
        encoding = self.tokenizer(text, **self.tokenizer_kwargs)

        # The tokenizer returns a dictionary with batch dimension 1, so we squeeze it.
        # The DataLoader will add the batch dimension back.
        inputs = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

        return inputs, torch.tensor(label, dtype=torch.long)