from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

def get_shared_components(df, device, input_tensor=True):

    # Extract and combine features
    rating_cols = [col for col in df.columns if "rating" in col]
    potential_cols = [col for col in df.columns if "potential" in col]
    team_record_cols = [
        "home_team_wins", "home_team_draws", "home_team_losses",
        "away_team_wins", "away_team_draws", "away_team_losses"
    ]
    bookmaker_cols = ["avg_home_prob", "avg_draw_prob", "avg_away_prob"]
    metadata_cols = ["stage", "league_id", "country_id"]
    features = (
        rating_cols +
        potential_cols +
        team_record_cols +
        bookmaker_cols +
        metadata_cols
    )

    # Handle missing values
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
