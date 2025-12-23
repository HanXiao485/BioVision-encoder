import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=512, num_classes=100):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.layers(self.flatten(x))