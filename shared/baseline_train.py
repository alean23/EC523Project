import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np

def train_model(model, X_train, y_train, optimizer, criterion, epochs=100, batch_size=32, log_every=10):
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0

        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"[Epoch {epoch}] Loss: {epoch_loss:.4f}")

def evaluate_model(model, X, y, batch_size=32):
    model.eval()
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for i in range(0, X.size(0), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(batch_y.cpu().numpy())
    
    all_true = np.array(all_true)
    all_preds = np.array(all_preds)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        all_true, all_preds, average=None, labels=[0, 1, 2]
    )
    
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_true, all_preds, average='macro'
    )
    
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_true, all_preds, average='weighted'
    )
    
    accuracy = (all_true == all_preds).mean()
    conf_matrix = confusion_matrix(all_true, all_preds, labels=[0, 1, 2])
    
    print("\nEvaluation Metrics:")
    print(f"{'Accuracy:':<15} {accuracy:.4f}")
    print(f"{'Macro Precision:':<15} {macro_precision:.4f}")
    print(f"{'Macro Recall:':<15} {macro_recall:.4f}")
    print(f"{'Macro F1:':<15} {macro_f1:.4f}")
    print(f"{'Weighted Precision:':<15} {weighted_precision:.4f}")
    print(f"{'Weighted Recall:':<15} {weighted_recall:.4f}")
    print(f"{'Weighted F1:':<15} {weighted_f1:.4f}")
    
    print("\nClass-wise Metrics:")
    print(f"{'Class':<6} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
        print(f"{i:<6} {p:.4f}      {r:.4f}      {f:.4f}      {s:<10}")
    
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'confusion_matrix': conf_matrix,
        'support': support
    }