import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train_model(model, X_train, y_train, optimizer, criterion, epochs=100, batch_size=32, log_every=10):

    # Set to train, accummulate states
    model.train()
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):

        # Randomly permute the data
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0
        correct = 0
        total = 0

        for i in range(0, X_train.size(0), batch_size):

            # Extract batch
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]

            # Get outputs, calculate loss, backprop
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward()
            optimizer.step()
            
            # Record statistics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        epoch_loss /= len(X_train) / batch_size
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Print updates
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

def evaluate_model(model, X, y, bookie_probs=None, initial_bankroll=1000, bet_fraction=0.01, 
                  batch_size=32, match_dates=None):

    # Set to eval, set arrays for stats
    model.eval()
    all_preds = []
    all_true = []
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, X.size(0), batch_size):

            # Extract batch
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            outputs = model(batch_X)

            # Get predictions
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Add them to stats
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_true = np.array(all_true)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Classification metrics
    classification_results = calculate_classification_metrics(all_true, all_preds)
    
    # KL Div if given bookie stats
    kl_results = None
    if bookie_probs is not None:
        kl_results = calculate_kl_divergence(all_probs, bookie_probs)
    
    # Run profit simulation
    profit_results = None
    if bookie_probs is not None:
        profit_results = simulate_profits(
            true_outcomes=all_true,
            model_probs=all_probs,
            bookie_probs=bookie_probs,
            initial_bankroll=initial_bankroll,
            bet_fraction=bet_fraction,
            match_dates=match_dates
        )
    
    results = {
        'classification': classification_results,
        'kl_divergence': kl_results,
        'profit_simulation': profit_results
    }
    
    return results

def calculate_classification_metrics(true_labels, pred_labels):

    # Yay scikit-learn
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, average=None, labels=[0, 1, 2]
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='macro'
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='weighted'
    )
    
    accuracy = (true_labels == pred_labels).mean()
    conf_matrix = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2])
    
  # Print everything
    print("\n=== Classification Metrics ===")
    print("Accuracy:", accuracy)
    print("Macro Precision:", macro_precision)
    print("Macro Recall:", macro_recall)
    print("Macro F1:", macro_f1)
    print("Weighted Precision:", weighted_precision)
    print("Weighted Recall:", weighted_recall)
    print("Weighted F1:", weighted_f1)

    print("\nClass-wise Metrics:")
    print("Class Precision Recall F1 Support")
    for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
        print(i, p, r, f, s)

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

def calculate_kl_divergence(model_probs, bookie_probs):

    # Avoid log 0
    epsilon = 1e-10
    model_probs_safe = np.clip(model_probs, epsilon, 1 - epsilon)
    bookie_probs_safe = np.clip(bookie_probs, epsilon, 1 - epsilon)
    
    # KL(P_bookie || P_model)
    kl_div = np.mean(np.sum(bookie_probs_safe * np.log(bookie_probs_safe / model_probs_safe), axis=1))
    
    print("\n=== Probability Distribution Metrics ===")
    print(f"{'KL Divergence (Bookie || Model):'} {kl_div}")
    
    return {
        'kl_divergence': kl_div
    }

def simulate_profits(true_outcomes, model_probs, bookie_probs, initial_bankroll=1000, 
                    bet_fraction=0.01, match_dates=None):

    bankroll = initial_bankroll
    bankroll_history = [bankroll]
    bets_made = 0
    bets_won = 0
    # Tuples of data, bankroll
    bet_records = []
    
    # Convert probs to odds
    bookie_odds = 1 / bookie_probs
    
    for i in range(len(true_outcomes)):

        # Find value bets
        value_bets = model_probs[i] > bookie_probs[i]
        
        if not any(value_bets):
            # Record no change in bankroll if no value bets found
            bet_records.append((match_dates[i], bankroll))
            continue
            
        for outcome in np.where(value_bets)[0]:

            # Calculate stake using Kelly criterion
            p = model_probs[i, outcome]
            b = bookie_odds[i, outcome] - 1
            f = (p * (b + 1) - 1) / b
            stake = bankroll * bet_fraction * f
            
            # Too small, don't bet
            if stake < 1:
                continue
            
            # Check outcome, update bankroll
            bets_made += 1
            if true_outcomes[i] == outcome:
                bankroll += stake * b
                bets_won += 1
            else:
                bankroll -= stake
            bet_records.append((match_dates[i], bankroll))
    
    # Calculate metrics
    total_return = bankroll - initial_bankroll
    roi = (total_return / initial_bankroll) * 100
    win_rate = bets_won / bets_made if bets_made > 0 else 0
    
    print("\n=== Profit Simulation Results ===")
    print(f"{'Initial Bankroll:'} ${initial_bankroll}")
    print(f"{'Final Bankroll:'} ${bankroll}")
    print(f"{'Total Profit/Loss:'} ${total_return}")
    print(f"{'ROI:'} {roi}%")
    print(f"{'Bets Made:'} {bets_made}")
    print(f"{'Bets Won:'} {bets_won}")
    print(f"{'Win Rate:'} {win_rate}")
    
    # Plotting profit
    if match_dates is not None and bet_records:
        plt.figure(figsize=(12, 6))
        
        df = pd.DataFrame(bet_records, columns=['date', 'balance'])
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        df = df.drop_duplicates('date', keep='last')
        df = df.sort_values('date')
        plt.plot(df['date'], df['balance'], label='Bankroll')
        plt.axhline(y=initial_bankroll, color='r', linestyle='--', label='Initial Bankroll')
        plt.xlabel('Date')
        plt.ylabel('Bankroll ($)')
        plt.title('Betting Performance Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('betting_performance.png', dpi=300, bbox_inches='tight')

    return {
        'final_bankroll': bankroll,
        'total_profit': total_return,
        'roi': roi,
        'bets_made': bets_made,
        'bets_won': bets_won,
        'win_rate': win_rate,
        'bankroll_history': bankroll_history,
        'bet_records': bet_records
    }