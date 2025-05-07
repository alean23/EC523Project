# Modelling Soccer Match Odds with Recurrent Neural Network Architectures


This project aims to predict the odds of a home win, draw, or away win for soccer matches based on team records and player/team statistics (drawn from FIFA).

In this repository, we've implemented and tested various RNNs on sequential data from the Big 5 soccer eagues.

## Data

We use this dataset: https://www.kaggle.com/datasets/hugomathien/soccer

Final generated dataset is in the final_matches_with_records.csv for madrid dataset

Final generated dataset is in final_combined_with_team_records.csv for the entire laliga dataset.

Data transformation occurs in madrid_data_creation.ipynb for the madrid dataset.

Data transformation occurs in laliga_data_creation.ipynb for the league-wide dataset.

## Approach

We've developed 5 models: 

1. Feed Forward Baseline
2. Bidirectional LSTM
3. Gated Recurrent Unit
4. Mamba
5. Custom Teamwise Stacked LSTM (assigns a single LSTM to each team and uses concatenated hidden states to predict match outcomes).

The code for each model can be found in its folder. Shared code for preprocessing and testing can be found in the shared/ folder.

Instructions to run and use code:

You will need to install PyTorch to run any of the models. For the Mamba model you will need to install mamba: https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html

To run each of the models, simply run the corresponding file in the folder for that model. Each file will train the model, and run the same testing criteria on each model.

## Testing Criteria:

1. Accuracy
2. F1 Score
3. KL divergence between predicted odds and average sportsbook odds
4. Kelly Criteria Betting Performance (Attempts to bet on testing data using an optimal strategy and measures ROI).

## Model Performance

<img width="329" alt="image" src="https://github.com/user-attachments/assets/f64c8c23-7119-4101-a972-ee64a782b89c" />

The teamwise stacked LSTM gives the best ROI/betting performance.



