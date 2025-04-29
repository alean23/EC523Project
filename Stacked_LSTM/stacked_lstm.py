import torch
import torch.nn as nn
import pandas as pd
import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.baseline_train import calculate_kl_divergence, calculate_classification_metrics, simulate_profits

import matplotlib.pyplot as plt


class StackedLSTMs(nn.Module):
    #potential todos: try implementing dropout, multiple lstm layers, bidirectional?
    def __init__(self, num_teams, input_size, hidden_size, output_size, device, initial_hx = None, initial_cx = None):
        super(StackedLSTMs, self).__init__()
        self.device = device
        self.num_teams = num_teams
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size, hidden_size, device = device)
        self.hx = initial_hx.to(device) if initial_hx else torch.randn(num_teams, hidden_size).to(device)
        self.cx = initial_cx.to(device) if initial_cx else torch.randn(num_teams, hidden_size).to(device)
        # One LSTM per team
        # self.lstm_layers = nn.ModuleList([
        #     nn.LSTMCell(input_size, hidden_size, device = device) for _ in range(num_teams)
        # ])
        # Output layer: takes concatenated hidden states of two teams
        self.output_layer = nn.Linear(hidden_size * 2, output_size).to(device)

    #For one match day
    def forward(self, inputs, matches):
        # inputs: list of tensors (num_teams, input_size)
        # matches: list of (home idx, away idx)        
        if matches[0].sum() == 0:
            return torch.Tensor([])
        outputs = []
        
        #Need temp variables since inplace operations mess up backprop
        new_hx = self.hx.clone()
        new_cx = self.cx.clone()
        for i in range(self.num_teams):
            new_hx[i], new_cx[i] = self.lstm(inputs[i], (self.hx[i], self.cx[i]))
            
        self.hx = new_hx.detach()
        self.cx = new_cx.detach()
        for home_idx, away_idx in matches:
            if home_idx + away_idx == 0:
                break
            logits = self.output_layer(torch.cat((self.hx[home_idx], self.hx[away_idx])))
            pmf = torch.softmax(logits, dim=0)
            outputs.append(pmf)
        return torch.stack(outputs)
    
def get_data(df, device):
    """
    Splits the DataFrame into train and test sets based on the 'season' column.
    Seasons '2008/2009' to '2013/2014' go to train, '2014/2015' and '2015/2016' go to test.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_columns (list): List of columns to use as features (X).
        target_column (str): Column to use as target (y).
        
    Returns:
        X_train, X_test, y_train, y_test (all pd.DataFrame or pd.Series)
    """
    def get_result_class(row):
        if row["home_team_goal"] > row["away_team_goal"]:
            return 0  # home win
        elif row["home_team_goal"] == row["away_team_goal"]:
            return 1  # draw
        else:
            return 2  # away win
        

        
    #Only teams that play in every single season
    allowed_teams = [
        'Valencia CF', 'RCD Espanyol', 'Athletic Club de Bilbao',
        'Atlético Madrid', 'Sevilla FC', 'Real Madrid CF',
        'FC Barcelona', 'Getafe CF', 'Málaga CF'
    ]
    
    df = df[
        df['home_team_name'].isin(allowed_teams) |
        df['away_team_name'].isin(allowed_teams)
    ].reset_index(drop=True)
        

        
    train_seasons = ['2008/2009', '2009/2010', '2010/2011', '2011/2012', '2012/2013', '2013/2014']
    test_seasons = ['2014/2015', '2015/2016']
    
    teams = sorted(allowed_teams)
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    num_teams = len(teams)
    
    home_feature_cols = [f'home_player_{i}_rating' for i in range(1, 12)] + \
                        [f'home_player_{i}_potential' for i in range(1, 12)] + \
                        ['home_team_wins', 'home_team_draws', 'home_team_losses']
    away_feature_cols = [f'away_player_{i}_rating' for i in range(1, 12)] + \
                        [f'away_player_{i}_potential' for i in range(1, 12)] + \
                        ['away_team_wins', 'away_team_draws', 'away_team_losses']
    
    feature_dim = len(home_feature_cols)  # Should be same as away_feature_cols

    num_matchdays = 38
    train_seq_len = len(train_seasons) * num_matchdays
    test_seq_len = len(test_seasons) * num_matchdays
             
    
    X_train = np.zeros((train_seq_len, num_teams, feature_dim), dtype=np.float32)
    X_test = np.zeros((test_seq_len, num_teams, feature_dim), dtype=np.float32)

    y_train = [[] for _ in range(train_seq_len)]
    y_test = [[] for _ in range(test_seq_len)]
    match_schedule_train = [[] for _ in range(train_seq_len)]
    match_schedule_test = [[] for _ in range(test_seq_len)]
    
    actual_results = []

                        
    def get_seq_idx(season, stage, is_train=True, num_matchdays=38):
        if is_train:
            season_num = train_seasons.index(season)
        else:
            season_num = test_seasons.index(season)
        return season_num * num_matchdays + (int(stage) - 1)  # stage is 1-based
    
    
    for idx, row in df.iterrows():
        season = row['season']
        stage = row['stage']
        if pd.isnull(season) or pd.isnull(stage):
            continue

        if season in train_seasons:
            seq_idx = get_seq_idx(season, stage, is_train=True)
            X = X_train
            y = y_train
            schedule = match_schedule_train
        elif season in test_seasons:
            seq_idx = get_seq_idx(season, stage, is_train=False)
            X = X_test
            y = y_test
            schedule = match_schedule_test
        else:
            continue  # skip seasons not in train or test

        home_team = row['home_team_name']
        away_team = row['away_team_name']
        if pd.isnull(home_team) or pd.isnull(away_team):
            continue

        if home_team in team_to_idx:
            home_idx = team_to_idx[home_team]
            home_features = row[home_feature_cols].astype(float).fillna(0.0).to_numpy(dtype=np.float32)
            X[seq_idx, home_idx, :] = home_features
        if away_team in team_to_idx:
            away_idx = team_to_idx[away_team]
            away_features = row[away_feature_cols].astype(float).fillna(0.0).to_numpy(dtype=np.float32)
            X[seq_idx, away_idx, :] = away_features

    
        if (home_team in team_to_idx) and (away_team in team_to_idx):
            home_idx = team_to_idx[home_team]
            away_idx = team_to_idx[away_team]
            schedule[seq_idx].append((home_idx, away_idx))
            odds = row[['avg_home_prob', 'avg_draw_prob', 'avg_away_prob']].astype(float).fillna(0.0).to_numpy(dtype=np.float32)
            y[seq_idx].append(odds)
            if season in test_seasons:
                actual_results.append(get_result_class(row))

     # Convert y and schedule to arrays of consistent shapes (pad with zeros if needed)

    
    max_matches_train = max(len(day) for day in y_train)
    max_matches_test = max(len(day) for day in y_test)
    
    y_train_arr = np.zeros((train_seq_len, max_matches_train, 3), dtype=np.float32)
    y_test_arr = np.zeros((test_seq_len, max_matches_test, 3), dtype=np.float32)
    schedule_train_arr = np.zeros((train_seq_len, max_matches_train, 2), dtype=np.int32)
    schedule_test_arr = np.zeros((test_seq_len, max_matches_test, 2), dtype=np.int32)
    
    for i, (ylist, slist) in enumerate(zip(y_train, match_schedule_train)):
        for j, (odds, match) in enumerate(zip(ylist, slist)):
            y_train_arr[i, j, :] = odds
            schedule_train_arr[i, j, :] = match
    for i, (ylist, slist) in enumerate(zip(y_test, match_schedule_test)):
        for j, (odds, match) in enumerate(zip(ylist, slist)):
            y_test_arr[i, j, :] = odds
            schedule_test_arr[i, j, :] = match


    return {
        'X_train': torch.from_numpy(X_train).to(device),
        'y_train': torch.from_numpy(y_train_arr).to(device),
        'match_schedule_train': torch.from_numpy(schedule_train_arr).to(device),
        'X_test': torch.from_numpy(X_test).to(device),
        'y_test': torch.from_numpy(y_test_arr).to(device),
        'match_schedule_test': torch.from_numpy(schedule_test_arr).to(device),
        'team_to_idx': team_to_idx,
        'teams': teams,
        'true_results': np.array(actual_results)
    }
    


#X_train: (seq_len, num_teams, input_size)
#match_schedule (seq_len, num_matches) - (home idx, away idx)
#y_train: (seq_len, matches, output_size) #Correct odds (or one hot of correct result)
def train_model(model, X_train, match_schedule, y_train, optimizer, criterion, epochs=100, batch_size=32, log_every=10):
    model.train()
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        all_outputs = []
        all_targets = []
        
        optimizer.zero_grad()


        for i in range(0, X_train.size(0)):
            #No matches between selected teams at this matchday
            if match_schedule[i][0].sum() == 0:
                continue
            
            outputs = model(X_train[i], match_schedule[i])
            targets = y_train[i][:outputs.shape[0]]
            
            all_outputs.append(outputs)
            all_targets.append(targets)

            
            _, predicted = torch.max(outputs.data, 1)
            total += outputs.shape[0]
            _, real = torch.max(y_train[i][:outputs.shape[0]], 1)

            correct += (predicted == real).sum().item()
            
        if len(all_outputs) > 0:
            combined_outputs = torch.cat(all_outputs)
            combined_targets = torch.cat(all_targets)
            loss = criterion(combined_outputs, combined_targets)
            loss.backward() 
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss = loss.item()
            
        epoch_loss = total_loss / len(X_train) if len(X_train) > 0 else 0
        epoch_acc = correct / total if total > 0 else 0
        
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"[Epoch {epoch}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    
    return train_losses, train_accuracies


def evaluate_model(model, X, y, match_schedule, all_true=None, initial_bankroll=1000, bet_fraction=0.01, 
                  batch_size=32, match_dates=None):
    model.eval()
    all_preds = []
    bookie_probs = []
    all_probs = []
    
    
    with torch.no_grad():
        for i in range(0, X.size(0)):
            outputs = model(X[i], match_schedule[i])
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            bookie_probs.extend(y[i][:len(predicted)].cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
    
    all_true = np.array(all_true)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate standard classification metrics

    classification_results = calculate_classification_metrics(all_true, all_preds)
    
    # Calculate KL divergence if bookie probabilities are provided
    kl_results = None
    
    if bookie_probs is not None:
        kl_results = calculate_kl_divergence(all_probs, bookie_probs)
    
    # Run profit simulation if bookie probabilities are provided
    profit_results = None
    if bookie_probs is not None:
        profit_results = simulate_profits(
            true_outcomes=all_true,
            model_probs=all_probs,
            bookie_probs=np.array(bookie_probs),
            initial_bankroll=initial_bankroll,
            bet_fraction=bet_fraction,
            match_dates=match_dates
        )
    
    # Combine all results
    results = {
        'classification': classification_results,
        'kl_divergence': kl_results,
        'profit_simulation': profit_results
    }
    
    return results

df = pd.read_csv("final_combined_with_team_records.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
data = get_data(df, device)


num_teams = len(data["teams"])  # La Liga teams
input_size = 25  # Feature vector size per matchday
hidden_size = 64
output_size = 3  # odds of win/draw/loss

model = StackedLSTMs(num_teams, input_size, hidden_size, output_size, device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
train_losses, train_accuracies = train_model(model, data["X_train"], data["match_schedule_train"], data["y_train"], optimizer, criterion)

test_seasons = ['2014/2015', '2015/2016']
# Convert date and create clean copies to avoid warnings
df = df.copy() 
df['match_date'] = pd.to_datetime(df['date'])
df = df[df['match_date'].notna()].copy()  
test_dates = df[df['season'].isin(test_seasons)]['match_date'].values

results = evaluate_model(model, data["X_test"], data["y_test"], data["match_schedule_test"], data["true_results"], match_dates=test_dates)