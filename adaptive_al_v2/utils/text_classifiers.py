import torch
from torch import nn


class SimpleTextClassifier(nn.Module):
    """Simple classifier for 3-feature text input (length, words, chars)"""

    def __init__(self, input_dim=3, hidden_dim=32, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, features):
        """
        Args:
            features: Tensor of shape (batch_size, 3)
                     [text_length, num_words, num_chars]
        """
        x = torch.relu(self.fc1(features))
        x = self.dropout(x)
        return self.fc2(x)