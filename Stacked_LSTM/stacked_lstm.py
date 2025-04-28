import torch
import torch.nn as nn
import pandas as pd

import sys
import os

from shared.baseline_shared import get_shared_components
from shared.baseline_train import train_model, evaluate_model

class StackedLSTMs(nn.Module):
    #potential todos: try implementing dropout, multiple lstm layers, bidirectional?
    def __init__(self, num_teams, input_size, hidden_size, device, initial_hx = None, initial_cx = None):
        super(StackedLSTMs, self).__init__()
        self.device = device
        self.num_teams = num_teams
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size, hidden_size, device = device)
        self.hx = initial_hx if initial_hx else torch.randn(num_teams, hidden_size)
        self.cx = initial_cx if initial_cx else torch.randn(num_teams, hidden_size)
        # One LSTM per team
        # self.lstm_layers = nn.ModuleList([
        #     nn.LSTMCell(input_size, hidden_size, device = device) for _ in range(num_teams)
        # ])
        # Output layer: takes concatenated hidden states of two teams
        self.output_layer = nn.Linear(hidden_size * 2, output_size)

    #For one match day
    def forward(self, inputs, matches):
        # inputs: list of tensors (num_teams, input_size)
        # matches: list of (home idx, away idx)
        outputs = []
        for i in range(self.num_teams):
            self.hx[i], self.cx[i] = self.lstm(inputs[i], (self.hx[i], self.cx[i]))
        for home_idx, away_idx in matches:
            logits = self.output_layer(torch.cat((self.hx[home_idx].squeeze(0), self.hx[away_idx].squeeze(0)), dim=1))
            pmf = torch.softmax(logits, dim=1)
            outputs.append(pmf)
        return outputs




# Example usage:
num_teams = 20  # La Liga teams
input_size = 50  # Feature vector size per matchday
hidden_size = 64
output_size = 3  # win/draw/loss

model = ParallelLSTMs(num_teams, input_size, hidden_size, output_size)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv("../final_combined_with_team_records.csv")

X_train, X_test, y_train, y_test, criterion, make_optimizer, input_dim = get_shared_components(df, device)

print(X_train)
print(y_train)
