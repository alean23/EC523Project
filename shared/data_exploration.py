import pandas as pd

df = pd.read_csv('final_combined_with_team_records.csv')

unique_teams = df['home_team_name'].unique()

print("Number of unique teams: ", len(unique_teams))
print(unique_teams)

#Teams that played every season (no relegation)
team_counts = df['home_team_name'].value_counts()
#A team plays 19 home games per season, so 19*8=152
teams_all_seasons = team_counts[team_counts == 152].index.tolist()

print("Teams playing every matchday of every season:")
print(teams_all_seasons)

#Teams that were in both test seasons and 4/6 train seasons
test_seasons = ['2014/2015', '2015/2016']
train_seasons = [
    '2008/2009', '2009/2010', '2010/2011',
    '2011/2012', '2012/2013', '2013/2014'
]

# Find teams that played as home in BOTH of the last two seasons
teams_2014_15 = set(df[df['season'] == '2014/2015']['home_team_name'])
teams_2015_16 = set(df[df['season'] == '2015/2016']['home_team_name'])
teams_in_both_last_two = teams_2014_15 & teams_2015_16

# For previous 6 seasons, count in how many unique seasons each team played as home
df_prev6 = df[df['season'].isin(train_seasons)]
team_season_counts = df_prev6.groupby('home_team_name')['season'].nunique()
teams_in_4_of_6 = set(team_season_counts[team_season_counts >= 4].index)

# Final teams: in both last two seasons AND at least 4 of previous 6
final_teams = sorted(list(teams_in_both_last_two & teams_in_4_of_6))

print("Teams playing in both test seasons and 4/6 train seasons:")
print(final_teams)

