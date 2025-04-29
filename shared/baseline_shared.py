from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

def get_shared_components(df, device, input_tensor=True, test_year=2016):
    # Define target function
    def get_result_class(row):
        if row["home_team_goal"] > row["away_team_goal"]:
            return 0  # home win
        elif row["home_team_goal"] == row["away_team_goal"]:
            return 1  # draw
        else:
            return 2  # away win

    # Convert date and create clean copies to avoid warnings
    df = df.copy() 
    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df[df['match_date'].notna()].copy()  
    
    # Temporal split with explicit copies
    train = df[df['match_date'].dt.year < test_year].copy()
    test = df[df['match_date'].dt.year == test_year].copy()
    
    # Extract features safely
    player_rating_cols = [col for col in df.columns if "rating" in col and "player" in col]
    player_potential_cols = [col for col in df.columns if "potential" in col and "player" in col]
    
    # Verify player stats
    assert len(player_rating_cols) == 22, f"Expected 22 player ratings, got {len(player_rating_cols)}"
    assert len(player_potential_cols) == 22, f"Expected 22 player potentials, got {len(player_potential_cols)}"
    
    # Other features
    team_record_cols = [
        "home_team_wins", "home_team_draws", "home_team_losses",
        "away_team_wins", "away_team_draws", "away_team_losses"
    ]
    metadata_cols = ["stage", "league_id"]
    features = player_rating_cols + player_potential_cols + team_record_cols + metadata_cols
    
    # Handle missing values safely
    median_values = train[features].median()
    train.loc[:, features] = train[features].fillna(median_values)
    test.loc[:, features] = test[features].fillna(median_values)
    
    # Bookmaker probabilities
    bookmaker_cols = ["avg_home_prob", "avg_draw_prob", "avg_away_prob"]
    bookie_probs_train = train[bookmaker_cols].values
    bookie_probs_test = test[bookmaker_cols].values
    
    # Create targets
    y_train = train.apply(get_result_class, axis=1)
    y_test = test.apply(get_result_class, axis=1)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    X_test = scaler.transform(test[features])
    
    # Convert to tensors
    if input_tensor:
        X_train = torch.FloatTensor(X_train).to(device)
        X_test = torch.FloatTensor(X_test).to(device)
        y_train = torch.LongTensor(y_train.values).to(device)
        y_test = torch.LongTensor(y_test.values).to(device)
        bookie_train = torch.FloatTensor(bookie_probs_train).to(device)
        bookie_test = torch.FloatTensor(bookie_probs_test).to(device)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'bookie_train': bookie_train,
        'bookie_test': bookie_test,
        'criterion': nn.CrossEntropyLoss(),
        'make_optimizer': lambda m: torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-5),
        'input_dim': len(features),
        'feature_names': features 
    }