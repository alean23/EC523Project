import torch
import torch.nn as nn
import pandas as pd
from mamba_ssm import Mamba

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.baseline_shared import get_shared_components
from shared.baseline_train import train_model, evaluate_model

class MatchPredictor(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.input_dim = input_dim
        
        # Calculate feature dimensions based on your preprocessing
        # Each team has 11 players × 2 features (rating + potential)
        self.num_players = 11
        self.player_feat_dim = 2  # rating + potential per player
        self.team_feat_dim = self.num_players * self.player_feat_dim  # 22
        
        # Verify we have enough features for both teams
        assert input_dim >= 2 * self.team_feat_dim, \
            f"Input dim {input_dim} too small for {2*self.team_feat_dim} team features"
        
        # Team projections (each team -> 64-dim)
        self.home_proj = nn.Linear(self.team_feat_dim, 64)
        self.away_proj = nn.Linear(self.team_feat_dim, 64)
        
        # Mamba layer (processes sequence of 2 elements - home and away teams)
        self.mamba = Mamba(
            d_model=64,  # Must match projection output
            d_state=16,
            d_conv=4,
            expand=2
        )
        
        # Remaining features (match context)
        self.match_context_dim = input_dim - (2 * self.team_feat_dim)
        self.match_context = nn.Sequential(
            nn.Linear(self.match_context_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classifier (64*2 from teams + 64 from context)
        self.classifier = nn.Sequential(
            nn.Linear(64*3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Split input into team features and context
        home_team = x[:, :self.team_feat_dim]
        away_team = x[:, self.team_feat_dim:2*self.team_feat_dim]
        match_context = x[:, 2*self.team_feat_dim:]
        
        # Project each team
        home_encoded = self.home_proj(home_team)
        away_encoded = self.away_proj(away_team)
        
        # Process as sequence (shape [batch, 2, 64])
        team_seq = torch.stack([home_encoded, away_encoded], dim=1)
        team_features = self.mamba(team_seq)
        
        # Average sequence outputs
        team_features = team_features.mean(dim=1)
        
        # Process context
        context_features = self.match_context(match_context)
        
        # Combine all features
        combined = torch.cat([
            home_encoded,  
            away_encoded,
            context_features
        ], dim=1)
        
        return self.classifier(combined)

# Usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv("../final_5_league.csv")
components = get_shared_components(df, device)

model = MatchPredictor(input_dim=components['input_dim']).to(device)
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

