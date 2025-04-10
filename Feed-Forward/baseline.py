import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.baseline_shared import get_shared_components
from shared.baseline_train import train_model, evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv("../final_combined_with_team_records.csv")

X_train, X_test, y_train, y_test, criterion, make_optimizer, input_dim = get_shared_components(df, device)


class BasicFeedForward(nn.Module):
    def __init__(self, input_dim):
        super(BasicFeedForward, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=1)  
        )

    def forward(self, x):
        return self.model(x)

model = BasicFeedForward(input_dim=input_dim).to(device)
optimizer = make_optimizer(model)

train_model(model, X_train, y_train, optimizer, criterion, epochs=200)
evaluate_model(model, X_test, y_test)
