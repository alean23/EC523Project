from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

def get_shared_components(df, device, input_tensor=True):
    # Extract player names and create mappings
    player_name_cols = [col for col in df.columns if 'player' in col and 'name' in col]
    all_player_names = pd.unique(df[player_name_cols].values.ravel())
    all_player_names = [name for name in all_player_names if pd.notna(name)]
    player_to_id = {name: idx for idx, name in enumerate(all_player_names)}

    # Create player ID columns
    for col in player_name_cols:
        id_col = col.replace('_name', '_id')
        df[id_col] = df[col].map(player_to_id)

    # Feature groups (EXCLUDING bookmaker odds)
    rating_cols = [col for col in df.columns if "rating" in col]
    potential_cols = [col for col in df.columns if "potential" in col]
    team_record_cols = [
        "home_team_wins", "home_team_draws", "home_team_losses",
        "away_team_wins", "away_team_draws", "away_team_losses"
    ]
    metadata_cols = ["stage", "league_id", "country_id"]
    player_id_cols = [col.replace('_name', '_id') for col in player_name_cols]

    # Features for training (no bookmaker odds)
    features = (
        rating_cols +
        potential_cols +
        team_record_cols +
        metadata_cols +
        player_id_cols
    )

    # Bookmaker probabilities for evaluation only
    bookmaker_cols = ["avg_home_prob", "avg_draw_prob", "avg_away_prob"]
    bookie_probs = df[bookmaker_cols].values

    # Handle missing values
    df[features] = df[features].fillna(df[features].median())

    # Create target classes
    def get_result_class(row):
        if row["home_team_goal"] > row["away_team_goal"]:
            return 0  # home win
        elif row["home_team_goal"] == row["away_team_goal"]:
            return 1  # draw
        else:
            return 2  # away win

    df["result"] = df.apply(get_result_class, axis=1)
    
    # Split data
    X = df[features]
    y = df["result"]
    X_train, X_test, y_train, y_test, bookie_train, bookie_test = train_test_split(
        X, y, bookie_probs, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if input_tensor:
        X_train = torch.FloatTensor(X_train).to(device)
        X_test = torch.FloatTensor(X_test).to(device)
        y_train = torch.LongTensor(y_train.to_numpy()).to(device)
        y_test = torch.LongTensor(y_test.to_numpy()).to(device)
        bookie_train = torch.FloatTensor(bookie_train).to(device)
        bookie_test = torch.FloatTensor(bookie_test).to(device)

    criterion = nn.CrossEntropyLoss()

    def make_optimizer(model):
        return torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'bookie_train': bookie_train,
        'bookie_test': bookie_test,
        'criterion': criterion,
        'make_optimizer': make_optimizer,
        'input_dim': X.shape[1],
        'player_feature_cols': rating_cols + potential_cols,
        'team_feature_cols': team_record_cols,
        'metadata_cols': metadata_cols
    }