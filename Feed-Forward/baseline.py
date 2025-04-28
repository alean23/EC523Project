import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.baseline_shared import get_shared_components
from shared.baseline_train import train_model, evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv("./final_5_league.csv")

X_train, X_test, y_train, y_test, criterion, make_optimizer, input_dim = get_shared_components(df, device)

# Define the model
class MyModel(nn.Module):
    def __init__(self, input_shape):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)  # 3 output classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)  # Softmax along classes
        return x
input_shape = X_train.shape[1]
model = MyModel(input_shape)

criterion = nn.CrossEntropyLoss()  # Note: expects class indices if labels are not one-hot
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, X_train, y_train, optimizer, criterion, epochs=200)
evaluate_model(model, X_test, y_test)
