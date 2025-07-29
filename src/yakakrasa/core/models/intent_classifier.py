import torch
import torch.nn as nn

class IntentClassifier(nn.Module):
    """
    A simple feed-forward neural network for intent classification.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(IntentClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Our featurizer produces a sparse matrix, so we convert it to dense
        if hasattr(x, "to_dense"):
            x = x.to_dense()
        
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.softmax(out)
        return out 