import torch
import torch.nn as nn
import pandas as pd

import sys
import os

from shared.baseline_shared import get_shared_components
from shared.baseline_train import train_model, evaluate_model

import matplotlib.pyplot as plt


class StackedLSTMs(nn.Module):
    #potential todos: try implementing dropout, multiple lstm layers, bidirectional?
    def __init__(self, num_teams, input_size, hidden_size, device, initial_hx = None, initial_cx = None):
        super(StackedLSTMs, self).__init__()
        self.device = device
        self.num_teams = num_teams
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size, hidden_size, device = device)
        self.hx = initial_hx if initial_hx else torch.randn(num_teams, hidden_size)
        self.cx = initial_cx if initial_cx else torch.randn(num_teams, hidden_size)
        # One LSTM per team
        # self.lstm_layers = nn.ModuleList([
        #     nn.LSTMCell(input_size, hidden_size, device = device) for _ in range(num_teams)
        # ])
        # Output layer: takes concatenated hidden states of two teams
        self.output_layer = nn.Linear(hidden_size * 2, output_size)

    #For one match day
    def forward(self, inputs, matches):
        # inputs: list of tensors (num_teams, input_size)
        # matches: list of (home idx, away idx)
        outputs = []
        for i in range(self.num_teams):
            self.hx[i], self.cx[i] = self.lstm(inputs[i], (self.hx[i], self.cx[i]))
        for home_idx, away_idx in matches:
            logits = self.output_layer(torch.cat((self.hx[home_idx].squeeze(0), self.hx[away_idx].squeeze(0)), dim=1))
            pmf = torch.softmax(logits, dim=1)
            outputs.append(pmf)
        return outputs
    
    #X
    def run_for_full_data(self, )

#X_train: (seq_len, num_teams, input_size)
#match_schedule (seq_len, num_matches) - (home idx, away idx)
#y_train: (seq_len, matches, output_size) #Correct odds (or one hot of correct result)
def train_model(model, X_train, match_schedule, y_train, optimizer, criterion, epochs=100, batch_size=32, log_every=10):
    model.train()
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for i in range(0, X_train.size(0)):
            optimizer.zero_grad()
            outputs = model(X_train[i], match_schedule[i])
            loss = criterion(outputs, y_train[i])

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_train[i].size(0)
            correct += (predicted == y_train[i]).sum().item()

        epoch_loss /= len(X_train) 
        epoch_acc = correct / total
        
        epoch_loss.backward()
        optimizer.step()

        
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

# Example usage:
num_teams = 20  # La Liga teams
input_size = 50  # Feature vector size per matchday
hidden_size = 64
output_size = 3  # win/draw/loss

model = StackedLSTMs(num_teams, input_size, hidden_size, output_size)
train_model(model, )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv("../final_combined_with_team_records.csv")

X_train, X_test, y_train, y_test, criterion, make_optimizer, input_dim = get_shared_components(df, device)

print(X_train)
print(y_train)
