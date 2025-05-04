from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

def get_shared_components(df, device, input_tensor=True, test_year=2016):
    # Class definitions
    def get_result_class(row):
        if row["home_team_goal"] > row["away_team_goal"]:
            # Home win
            return 0 
        elif row["home_team_goal"] == row["away_team_goal"]:
            # Draw
            return 1  
        else:
            # Away win
            return 2 

    # Date conversion
    df = df.copy() 
    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df[df['match_date'].notna()].copy()  
    
    # Temporal split on 2016
    train = df[df['match_date'].dt.year < test_year].copy()
    test = df[df['match_date'].dt.year == test_year].copy()
    
    
    # Extract features
    player_attr_cols = [col for col in df.columns 
                   if any(x in col for x in ['rating', 'potential', 'crossing', 'finishing', 
                                           'heading', 'passing', 'dribbling', 'curve', 
                                           'accuracy', 'control', 'acceleration', 'speed',
                                           'agility', 'reactions', 'balance', 'power',
                                           'jumping', 'stamina', 'strength', 'shots',
                                           'aggression', 'interceptions', 'positioning',
                                           'vision', 'penalties', 'marking', 'tackle',
                                           'diving', 'handling', 'kicking', 'positioning',
                                           'reflexes'])]
    
    # Other features
    team_record_cols = [
        "home_team_wins", "home_team_draws", "home_team_losses",
        "away_team_wins", "away_team_draws", "away_team_losses"
    ]
    metadata_cols = ["stage", "league_id"]
    features = player_attr_cols + team_record_cols + metadata_cols
    
    # Missing values
    median_values = train[features].median()
    train.loc[:, features] = train[features].fillna(median_values)
    test.loc[:, features] = test[features].fillna(median_values)
    
    # Bookmaker cols probabilities
    bookmaker_cols = ["avg_home_prob", "avg_draw_prob", "avg_away_prob"]
    bookie_probs_train = train[bookmaker_cols].values
    bookie_probs_test = test[bookmaker_cols].values
    
    # Targets
    y_train = train.apply(get_result_class, axis=1)
    y_test = test.apply(get_result_class, axis=1)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    X_test = scaler.transform(test[features])
    
    # Tensors
    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.LongTensor(y_train.values).to(device)
    y_test = torch.LongTensor(y_test.values).to(device)
    bookie_train = torch.FloatTensor(bookie_probs_train).to(device)
    bookie_test = torch.FloatTensor(bookie_probs_test).to(device)
    
    # Return common features
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