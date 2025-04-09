import torch
import torch.nn as nn
import pandas as pd
from mamba_ssm import Mamba

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.baseline_shared import get_shared_components
from shared.baseline_train import train_model, evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv("../final_combined_with_team_records.csv")

X_train, X_test, y_train, y_test, criterion, make_optimizer, input_dim = get_shared_components(df, device)

import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MatchPredictor(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.input_dim = input_dim

        self.num_players = 11
        self.player_feat_dim = 2
        self.team_feat_dim = self.num_players * self.player_feat_dim  # 22

        self.home_proj = nn.Linear(self.team_feat_dim, 64)
        self.away_proj = nn.Linear(self.team_feat_dim, 64)

        self.mamba = Mamba(
            d_model=64,
            d_state=16,
            d_conv=4,
            expand=2
        )

        self.match_context_dim = input_dim - (2 * self.team_feat_dim)
        
        self.match_context = nn.Sequential(
            nn.Linear(self.match_context_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64*2 + 64, 64), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        home_team = x[:, :self.team_feat_dim]
        away_team = x[:, self.team_feat_dim:2*self.team_feat_dim]
        match_context = x[:, 2*self.team_feat_dim:]
        home_encoded = self.home_proj(home_team)
        away_encoded = self.away_proj(away_team)
        team_seq = torch.stack([home_encoded, away_encoded], dim=1)
        team_features = self.mamba(team_seq)
        team_features = team_features.mean(dim=1) 
        
        context_features = self.match_context(match_context)
        
        combined = torch.cat([
            home_encoded, 
            away_encoded, 
            context_features 
        ], dim=1)
        
        return self.classifier(combined)

model = MatchPredictor(input_dim=input_dim).to(device)
optimizer = make_optimizer(model)

train_model(model, X_train, y_train, optimizer, criterion, epochs=200)
evaluate_model(model, X_test, y_test)
