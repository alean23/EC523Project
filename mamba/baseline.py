import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from mamba_ssm import Mamba
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
df = pd.read_csv("../final_matches_with_records.csv")

def prepare_features(df):
    features = [
        'real_wins', 'real_draws', 'real_losses',
        'other_wins', 'other_draws', 'other_losses',
        *[f'real_player_{i}_rating' for i in range(1, 12)],
        *[f'other_player_{i}_rating' for i in range(1, 12)],
        *[f'real_player_{i}_potential' for i in range(1, 6)],
        *[f'other_player_{i}_potential' for i in range(1, 6)],
    ]
    
    medians = df[features].median()
    df[features] = df[features].fillna(medians)
    
    df['real_avg_rating'] = df[[f'real_player_{i}_rating' for i in range(1, 12)]].mean(axis=1)
    df['other_avg_rating'] = df[[f'other_player_{i}_rating' for i in range(1, 12)]].mean(axis=1)
    df['real_avg_potential'] = df[[f'real_player_{i}_potential' for i in range(1, 6)]].mean(axis=1)
    df['other_avg_potential'] = df[[f'other_player_{i}_potential' for i in range(1, 6)]].mean(axis=1)
    
    X = df[features]
    y = df['result']
    
    return X.values, y.values

X, y = prepare_features(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.FloatTensor(X_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_train = torch.LongTensor(y_train).to(device)
y_test = torch.LongTensor(y_test).to(device)

class MatchPredictor(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.mamba = Mamba(
            d_model=64,
            d_state=16,
            d_conv=4,
            expand=2,
        ).to(device)
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        ).to(device)
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        ).to(device)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)  
        x = self.mamba(x)
        x = x.squeeze(1)   
        return self.classifier(x)

model = MatchPredictor(input_dim=X_train.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

def train(model, X, y, epochs=100, batch_size=32):
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X.size(0))
        epoch_loss = 0
        
        for i in range(0, X.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X[indices], y[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                _, predicted = torch.max(val_outputs, 1)
                correct = (predicted == y_test).sum().item()
                acc = correct / y_test.size(0)
            print(f'Epoch {epoch}, Loss: {epoch_loss:.4f}, Val Acc: {acc:.4f}')
            model.train()

train(model, X_train, y_train, epochs=200)