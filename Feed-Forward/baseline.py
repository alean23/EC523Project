import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.baseline_shared import get_shared_components
from shared.baseline_train import train_model, evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv("./final_5_league.csv")

# Define the model
class MyModel(nn.Module):
    def __init__(self, input_shape):
        super(MyModel, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ReLU(),

            nn.Linear(16, 3),  # Final layer, no dropout or activation
        )

    def forward(self, x):
        x = self.net(x)
        return torch.softmax(x, dim=1)  # Output probabilities
    
# Usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv("./final_5_league.csv")
components = get_shared_components(df, device)

model = MyModel(input_shape=components['input_dim']).to(device)
optimizer = components['make_optimizer'](model)

df['match_date'] = pd.to_datetime(df['match_date'])
test_dates = df[df['match_date'].dt.year == 2016]['match_date'].values

train_model(
    model,
    components['X_train'],
    components['y_train'],
    optimizer,
    components['criterion'],
    epochs=200
)

evaluate_model(
    model,
    components['X_test'],
    components['y_test'],
    bookie_probs=components['bookie_test'].cpu().numpy(),
    match_dates=test_dates
)
