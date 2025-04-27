from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import pandas as pd

def get_shared_components(df, device, input_tensor=True):
    player_name_cols = [col for col in df.columns if 'player' in col and 'name' in col]

    # Get all unique player names
    all_player_names = pd.unique(df[player_name_cols].values.ravel())
    all_player_names = [name for name in all_player_names if pd.notna(name)]

    # Create player name -> ID mapping
    player_to_id = {name: idx for idx, name in enumerate(all_player_names)}

    # Map player names to player IDs and create new ID columns
    for col in player_name_cols:
        id_col = col.replace('_name', '_id')  # e.g., home_player_1_name → home_player_1_id
        df[id_col] = df[col].map(player_to_id)

    # Step 2: Now extract and combine features as before

    rating_cols = [col for col in df.columns if "rating" in col]
    potential_cols = [col for col in df.columns if "potential" in col]
    team_record_cols = [
        "home_team_wins", "home_team_draws", "home_team_losses",
        "away_team_wins", "away_team_draws", "away_team_losses"
    ]
    bookmaker_cols = ["avg_home_prob", "avg_draw_prob", "avg_away_prob"]
    metadata_cols = ["stage", "league_id", "country_id"]
    player_id_cols = [col.replace('_name', '_id') for col in player_name_cols]  # NEW

    # Now include player IDs in features
    features = (
        rating_cols +
        potential_cols +
        team_record_cols +
        bookmaker_cols +
        metadata_cols +
        player_id_cols  # NEW
    )

    # Step 3: Handle missing values
    df[features] = df[features].fillna(df[features].median())
    # Set up target classes
    def get_result_class(row):
        if row["home_team_goal"] > row["away_team_goal"]:
            return 0  # home win
        elif row["home_team_goal"] == row["away_team_goal"]:
            return 1
        else:
            return 2

    df["result"] = df.apply(get_result_class, axis=1)
    X = df[features]
    y = df["result"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if input_tensor:
        X_train = torch.FloatTensor(X_train).to(device)
        X_test = torch.FloatTensor(X_test).to(device)
        y_train = torch.LongTensor(y_train.to_numpy()).to(device)
        y_test = torch.LongTensor(y_test.to_numpy()).to(device)

    criterion = nn.CrossEntropyLoss()

    def make_optimizer(model):
        return torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    return X_train, X_test, y_train, y_test, criterion, make_optimizer, X.shape[1]
