from torch.utils.data import Dataset
import torch

class TextDataset(Dataset):
    """Simple dataset"""

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def _text_to_tensor(self, text):
        """Convert text to a simple tensor representation"""
        # Simple approach: use text length and character count as features
        text_str = str(text)

        # Create simple features
        length = len(text_str)
        num_words = len(text_str.split())
        num_chars = len([c for c in text_str if c.isalpha()])

        # Return as tensor (3 features)
        return torch.tensor([length, num_words, num_chars], dtype=torch.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Convert text to tensor
        text_tensor = self._text_to_tensor(self.texts[idx])

        # Convert label to tensor
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return text_tensor, label_tensor