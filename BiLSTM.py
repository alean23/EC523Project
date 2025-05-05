# Import libraries
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

from shared.baseline_shared import get_shared_components
from shared.baseline_train import train_model, evaluate_model

# Load data from cloned repository
data = pd.read_csv('final_combined_with_team_records.csv').fillna(0)
print(f"Data loaded successfully. Shape: {data.shape}")

# Tokenize player names
player_name_columns = [col for col in data.columns if 'player' in col and 'name' in col]
tokenizer = Tokenizer(filters='', split='\t')
all_names = []
for col in player_name_columns:
    all_names.extend(data[col].fillna('').astype(str).values)
tokenizer.fit_on_texts(all_names)
for col in player_name_columns:
    data[col] = data[col].fillna('').astype(str).map(
        lambda name: tokenizer.texts_to_sequences([name])[0][0] if name else 0
    )

data = data.sort_values(by='date')

def enhanced_feature_engineering(df):
    df = df.copy()
    # Team strength ratios
    df['home_strength'] = df['home_team_wins'] / (df['home_team_wins'] + df['home_team_losses'] + 1e-5)
    df['away_strength'] = df['away_team_wins'] / (df['away_team_wins'] + df['away_team_losses'] + 1e-5)
   

    for team in ['home', 'away']:
        df[f'{team}_form'] = df[f'{team}_team_wins'].rolling(3).mean()
        df[f'{team}_goals_avg'] = df[f'{team}_team_goal'].rolling(3).mean()
    return df.fillna(0)

data = enhanced_feature_engineering(data)

selected_features = [
    'avg_home_prob', 'avg_draw_prob', 'avg_away_prob',
    'home_strength', 'away_strength',
    'home_form', 'away_form',
    'home_goals_avg', 'away_goals_avg',
    'home_player_1_rating', 'home_player_2_rating',
    'away_player_1_rating', 'away_player_2_rating'
]
features = data[selected_features]
y = data['outcome']

sequence_length = 5
X_sequences = []
y_sequences = []
for i in range(len(features) - sequence_length):
    seq = features.iloc[i:i+sequence_length].values
    label = y.iloc[i+sequence_length]
    X_sequences.append(seq)
    y_sequences.append(label)

X = np.array(X_sequences)
y = np.array(y_sequences)

# Train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize features
def scale_sequences(X_tr, X_v, X_te):
    scaler = StandardScaler()
    orig_shape = X_tr.shape
    X_tr_s = scaler.fit_transform(X_tr.reshape(-1, orig_shape[-1]))
    X_v_s  = scaler.transform(X_v.reshape(-1, orig_shape[-1]))
    X_te_s = scaler.transform(X_te.reshape(-1, orig_shape[-1]))
    return (
        X_tr_s.reshape(orig_shape),
        X_v_s.reshape(X_v.shape),
        X_te_s.reshape(X_te.shape)
    )

X_train, X_val, X_test = scale_sequences(X_train, X_val, X_test)


##########################################################################################################################
### CONNECTING ORIGINAL BiLSTM MODEL TO FULL PROJECT ###

# Define PyTorch BiLSTM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTMMatchPredictor(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_layers=2, dropout=0.2, n_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        feat = self.dropout(last)
        return self.classifier(feat)

if __name__ == "__main__":
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["match_date"] = df["date"]

    components = get_shared_components(df, device)

    # Instantiate, train, and evaluate
    model = BiLSTMMatchPredictor(
        input_dim=components["input_dim"],
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        n_classes=3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    weights = torch.tensor([1.0, 2.5, 1.8], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    history = train_model(
        model,
        components["X_train"],
        components["y_train"],
        optimizer,
        criterion,
        epochs=200,
        batch_size=32
    )

    test_dates = df[df["match_date"].dt.year == 2016]["match_date"].values
    evaluate_model(
        model,
        components["X_test"],
        components["y_test"],
        bookie_probs=components["bookie_test"].cpu().numpy(),
        match_dates=test_dates
    )