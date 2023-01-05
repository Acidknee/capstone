from sportsreference.nfl.boxscore import Boxscores, Boxscore
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score


#get the schedule for the games
def schedule():
    year = 2021
    schedule = pd.DataFrame()
    weeks = list(range(1,19))
    for i in range(len(weeks)):
        date = str(weeks[i]) +'-'+ str(year)
        scores = Boxscores(weeks[i], year)
        week_games = pd.DataFrame()
        for x in range(len(scores.games[date])):
            game = pd.DataFrame(scores.games[date][x], index = [0])[['home_name', 'away_name', 'winning_name']]
            game['week'] = weeks[i]
            week_games = pd.concat([week_games, game])
        schedule = pd.concat([schedule, week_games]).reset_index().drop(columns = 'index')

    return schedule, year

#extract statistic from games
def extract_stats(year):
    week_games = pd.DataFrame()  
    game_stats = pd.DataFrame()
    try:
        week = int(input('Enter a week between 3-18: '))
        if week > 18 or week < 3:
            week = 10
    except ValueError:
        week = 10
    
    game_data = pd.DataFrame()
    weeks = list(range(1, week + 1))
    for i in range(len(weeks)):
        date = str(weeks[i]) +'-'+ str(year)
        scores = Boxscores(weeks[i], year)
        week_games = pd.DataFrame()
        for x in range(len(scores.games[date])):
            game = scores.games[date][x]['boxscore']
            game_stats = Boxscore(game)
            game_stats_df = game_stats.dataframe.reset_index().drop(columns = 'index')
            game_df = pd.DataFrame(scores.games[date][x], index = [0])
            game_df['week'] = weeks[i]  
            #assign win/ loss values for each game to calculate percentage
            try:
                if game_df.loc[0, 'home_score'] > game_df.loc[0, 'away_score']:
                    game_df = pd.merge(game_df, pd.DataFrame({'home_won' : [1], 'home_lost' : [0]}),left_index = True, right_index = True)
                    game_df = pd.merge(game_df, pd.DataFrame({'away_won' : [0], 'away_lost' : [1]}),left_index = True, right_index = True)
                elif game_df.loc[0, 'home_score'] < game_df.loc[0, 'away_score']:
                     game_df = pd.merge(game_df, pd.DataFrame({'home_won' : [0], 'home_lost' : [1]}),left_index = True, right_index = True)
                     game_df = pd.merge(game_df, pd.DataFrame({'away_won' : [1], 'away_lost' : [0]}),left_index = True, right_index = True)
                else:
                    game_df = pd.merge(game_df, pd.DataFrame({'home_won' : [0], 'home_lost' : [0]}),left_index = True, right_index = True)
                    game_df = pd.merge(game_df, pd.DataFrame({'away_won' : [0], 'away_lost' : [0]}),left_index = True, right_index = True)
            except TypeError:
                    game_df = pd.merge(game_df, pd.DataFrame({'home_won' : [np.nan], 'home_lost' : [np.nan]}),left_index = True, right_index = True)
                    game_df = pd.merge(game_df, pd.DataFrame({'away_won' : [np.nan], 'away_lost' : [np.nan]}),left_index = True, right_index = True)
            game_df = pd.merge(game_df, game_stats_df, left_index = True, right_index = True)
            game_df['away_time_of_possession'] = (int(game_df['away_time_of_possession'].loc[0][1:2]) * 60) + int(game_df['away_time_of_possession'].loc[0][3:5])
            game_df['home_time_of_possession'] = (int(game_df['home_time_of_possession'].loc[0][1:2]) * 60) + int(game_df['home_time_of_possession'].loc[0][3:5])
            week_games = pd.concat([week_games, game_df])
        #concatenate all of the data into one data frame and drop useless columns
        game_data = pd.concat([game_data, week_games]).reset_index().drop(columns = 'index')
        game_data = game_data.drop(columns = ['boxscore', 'winning_name_x', 'winning_abbr_x', 'losing_name_x', 'losing_abbr_x',
                                      'attendance', 'time', 'winner', 'winning_abbr_y', 'winning_name_y', 'stadium', 'losing_name_y',
                                      'losing_abbr_y', 'date', 'duration'])
            
    return game_data, week

def aggregate_data(schedule, game_data, week):
    ind = list()
    length = len(game_data.index)
    game_data[['first_downs_dif', 'home_win_perc']] = ''
    #populate an aggregate data frame with the average of all the columns
    for i in range(0, length):
        ind.append(int(i))
        try:
            game_data['away_fourth_down_perc'].iloc[[i]] =game_data['away_fourth_down_conversions'].iloc[ind].mean() /game_data['away_fourth_down_attempts'].iloc[ind].mean()
            game_data['home_fourth_down_perc'].iloc[[i]] =game_data['home_fourth_down_conversions'].iloc[ind].mean() /game_data['home_fourth_down_attempts'].iloc[ind].mean()
            game_data['away_third_down_perc'].iloc[[i]] =game_data['away_third_down_conversions'].iloc[ind].mean() /game_data['away_third_down_attempts'].iloc[ind].mean()
            game_data['home_third_down_perc'].iloc[[i]] =game_data['home_third_down_conversions'].iloc[ind].mean() /game_data['home_third_down_attempts'].iloc[ind].mean()
        except ZeroDivisionError:
            game_data['away_fourth_down_perc'] = 0
            game_data['home_fourth_down_perc'] = 0
            game_data['away_third_down_perc'] = 0
            game_data['home_third_down_perc'] = 0
        game_data['away_fourth_down_perc'] =game_data['away_fourth_down_perc'].fillna(0)
        game_data['home_fourth_down_perc'] =game_data['home_fourth_down_perc'].fillna(0)
        game_data['away_third_down_perc'] =game_data['away_third_down_perc'].fillna(0)
        game_data['home_third_down_perc'] =game_data['home_third_down_perc'].fillna(0)
        
        try:
            game_data['home_win_per'] =game_data['home_won'].iloc[ind].sum() / (game_data['home_won'].iloc[ind].sum() +game_data['home_lost'].iloc[ind].sum())
            game_data['away_win_per'] =game_data['away_won'].iloc[ind].sum() / (game_data['away_won'].iloc[ind].sum() +game_data['away_lost'].iloc[ind].sum())
            game_data['win_perc_dif'] =game_data['home_win_per'] -game_data['away_win_per']   
            game_data['first_downs_dif'] =game_data['home_first_downs'].iloc[ind].mean(axis = 0) -game_data['away_first_downs'].iloc[ind].mean(axis = 0)
            game_data['fumbles_dif'] =game_data['home_fumbles'].iloc[ind].mean(axis = 0) -game_data['away_fumbles'].iloc[ind].mean(axis = 0)
            game_data['interceptions_dif'] =game_data['home_interceptions'].iloc[ind].mean(axis = 0) -game_data['away_interceptions'].iloc[ind].mean(axis = 0)
            game_data['net_pass_yards_dif'] = game_data['home_net_pass_yards'].iloc[ind].mean(axis = 0) -game_data['away_net_pass_yards'].iloc[ind].mean(axis = 0)
            game_data['pass_attempts_dif'] = game_data['home_pass_attempts'].iloc[ind].mean(axis = 0) -game_data['away_pass_attempts'].iloc[ind].mean(axis = 0)
            game_data['pass_completions_dif'] =game_data['home_pass_completions'].iloc[ind].mean(axis = 0) -game_data['away_pass_completions'].iloc[ind].mean(axis = 0)
            game_data['pass_touchdowns_dif'] =game_data['home_pass_touchdowns'].iloc[ind].mean(axis = 0) -game_data['away_pass_touchdowns'].iloc[ind].mean(axis = 0)
            game_data['pass_yards_dif'] =game_data['home_pass_yards'].iloc[ind].mean(axis = 0) -game_data['away_pass_yards'].iloc[ind].mean(axis = 0)
            game_data['penalties_dif'] =game_data['home_penalties'].iloc[ind].mean(axis = 0) -game_data['away_penalties'].iloc[ind].mean(axis = 0)
            game_data['points_dif'] =game_data['home_points'] -game_data['away_points']
            game_data['rush_attempts_dif'] =game_data['home_rush_attempts'] -game_data['away_rush_attempts']
            game_data['rush_touchdowns_dif'] =game_data['home_rush_touchdowns'] -game_data['away_rush_touchdowns']
            game_data['rush_yards_dif'] =game_data['home_rush_yards'] -game_data['away_rush_yards']
            game_data['time_of_possession_dif'] =game_data['home_time_of_possession'] -game_data['away_time_of_possession']
            game_data['times_sacked_dif'] =game_data['home_times_sacked'] -game_data['away_times_sacked']
            game_data['total_yards_dif'] =game_data['home_total_yards'] -game_data['away_total_yards']
            game_data['turnovers_dif'] =game_data['home_turnovers'] -game_data['away_turnovers']
            game_data['yards_from_penalties_dif'] =game_data['home_yards_from_penalties'] -game_data['away_yards_from_penalties']
            game_data['yards_lost_from_sacks_dif'] =game_data['home_yards_lost_from_sacks'] -game_data['away_yards_lost_from_sacks']
            game_data['fourth_down_perc_dif'] =game_data['home_fourth_down_perc'] -game_data['away_fourth_down_perc']
            game_data['third_down_perc_dif'] =game_data['home_third_down_perc'] -game_data['away_third_down_perc']       
        except ZeroDivisionError:
            game_data['home_win_per'] = 0
            game_data['away_win_per'] = 0
            game_data['win_perc_dif'] = 0
            
        game_data['win_perc_dif'] =game_data['win_perc_dif'].fillna(0) 
    
    game_data =game_data.drop(columns = ['away_score', 'home_score'])
    game_data = pd.merge(game_data, schedule, how = 'inner', left_on = ['home_name', 'away_name', 'week'], right_on = ['home_name', 'away_name', 'week'])
    game_data['result'] = game_data['winning_name'] == game_data['home_name']
    game_data['result'] = game_data['result'].astype('float')
    game_data =game_data.drop(columns = ['winning_name'])
    game_data =game_data.drop(game_data.loc[:, 'home_won':'away_win_per'].columns, axis = 1)
    
    return game_data

#retrieves the elo metric
def elo():
    elo_df = pd.read_csv('nfl_elo_latest.csv')
    elo_df = elo_df.drop(elo_df.loc[:, 'season': 'playoff'].columns, axis = 1)
    elo_df = elo_df.drop(elo_df.loc[:, 'elo_prob1': 'qb2'].columns, axis =1)
    elo_df = elo_df.drop(elo_df.loc[:, 'qb1_adj': 'score2'].columns, axis = 1)
    
    elo_df['team1'] = elo_df['team1'].replace(['KC', 'JAX', 'CAR', 'BAL', 'BUF', 'MIN', 'DET', 'ATL', 'NE', 'WSH',
           'CIN', 'NO', 'SF', 'LAR', 'NYG', 'DEN', 'CLE', 'IND', 'TEN', 'NYJ',
           'TB', 'MIA', 'PIT', 'PHI', 'GB', 'CHI', 'DAL', 'ARI', 'LAC', 'HOU',
           'SEA', 'OAK'],
            ['Kansas','Jacksonville','Carolina', 'Baltimore', 'Buffalo', 'Minnesota', 'Detroit', 'Atlanta', 'New England', 'Washington', 
            'Cincinnati', 'New Orleans', 'San Francisco', 'Los Angeles Rams', 'New York Giants', 'Denver', 'Cleveland', 'Indianapolis', 'Tennessee', 'New York Jets', 
             'Tampa Bay','Miami', 'Pittsburgh', 'Philadelphia', 'Green Bay', 'Chicago', 'Dallas', 'Arizona', 'Los Angeles Chargers', 'Houston', 'Seattle', 'Las Vegas' ])
    elo_df['team2'] = elo_df['team2'].replace(['KC', 'JAX', 'CAR', 'BAL', 'BUF', 'MIN', 'DET', 'ATL', 'NE', 'WSH',
           'CIN', 'NO', 'SF', 'LAR', 'NYG', 'DEN', 'CLE', 'IND', 'TEN', 'NYJ',
           'TB', 'MIA', 'PIT', 'PHI', 'GB', 'CHI', 'DAL', 'ARI', 'LAC', 'HOU',
           'SEA', 'OAK'],
            ['Kansas','Jacksonville','Carolina', 'Baltimore', 'Buffalo', 'Minnesota', 'Detroit', 'Atlanta', 'New England', 'Washington', 
            'Cincinnati', 'New Orleans', 'San Francisco', 'Los Angeles Rams', 'New York Giants', 'Denver', 'Cleveland', 'Indianapolis', 'Tennessee', 'New York Jets', 
             'Tampa Bay','Miami', 'Pittsburgh', 'Philadelphia', 'Green Bay', 'Chicago', 'Dallas', 'Arizona', 'Los Angeles Chargers', 'Houston', 'Seattle', 'Las Vegas' ])
    
    return elo_df

schedule, year = schedule()
game_data, week = extract_stats(year)
agg_data = aggregate_data(schedule, game_data, week)
elo_df = elo()

agg_data['home_abbr'] = agg_data['home_abbr'].replace(['kan','jax','car', 'rav', 'buf', 'min', 'det', 'atl', 'nwe', 'was', 
            'cin', 'nor', 'sfo', 'ram', 'nyg', 'den', 'cle', 'clt', 'oti', 'nyj', 
             'tam','mia', 'pit', 'phi', 'gnb', 'chi', 'dal', 'crd', 'sdg', 'htx', 'sea', 'rai'],
            ['Kansas','Jacksonville','Carolina', 'Baltimore', 'Buffalo', 'Minnesota', 'Detroit', 'Atlanta', 'New England', 'Washington', 
            'Cincinnati', 'New Orleans', 'San Francisco', 'Los Angeles Rams', 'New York Giants', 'Denver', 'Cleveland', 'Indianapolis', 'Tennessee', 'New York Jets', 
             'Tampa Bay','Miami', 'Pittsburgh', 'Philadelphia', 'Green Bay', 'Chicago', 'Dallas', 'Arizona', 'Los Angeles Chargers', 'Houston', 'Seattle', 'Las Vegas'])
agg_data['away_abbr'] = agg_data['away_abbr'].replace(['kan','jax','car', 'rav', 'buf', 'min', 'det', 'atl', 'nwe', 'was', 
            'cin', 'nor', 'sfo', 'ram', 'nyg', 'den', 'cle', 'clt', 'oti', 'nyj', 
             'tam','mia', 'pit', 'phi', 'gnb', 'chi', 'dal', 'crd', 'sdg', 'htx', 'sea', 'rai'],
            ['Kansas','Jacksonville','Carolina', 'Baltimore', 'Buffalo', 'Minnesota', 'Detroit', 'Atlanta', 'New England', 'Washington', 
            'Cincinnati', 'New Orleans', 'San Francisco', 'Los Angeles Rams', 'New York Giants', 'Denver', 'Cleveland', 'Indianapolis', 'Tennessee', 'New York Jets', 
             'Tampa Bay','Miami', 'Pittsburgh', 'Philadelphia', 'Green Bay', 'Chicago', 'Dallas', 'Arizona', 'Los Angeles Chargers', 'Houston', 'Seattle', 'Las Vegas'])


agg_data = pd.merge(agg_data, elo_df, how = 'inner', left_on = ['home_abbr', 'away_abbr'], right_on = ['team1', 'team2'])
agg_data = agg_data.drop(columns = ['date','team1', 'team2'])


agg_data['elo_dif'] = agg_data['elo2_pre'] - agg_data['elo1_pre']
agg_data['qb_dif'] = agg_data['qb2_value_pre'] - agg_data['qb1_value_pre']
agg_data = agg_data.drop(columns = ['elo1_pre', 'elo2_pre', 'qb1_value_pre', 'qb2_value_pre'])
    
agg_data = pd.read_csv('agg_data')

#seeting up data frames for logistic regression
agg_train = agg_data[agg_data.result.notna()]
agg_test = agg_data[agg_data.week == (week)]

model = np.random.rand(len(agg_train)) < .8


train = agg_train[model]
test = agg_train[~model]

x_train = train.drop(columns = ['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'result'])
y_train = train[['result']]
x_test = test.drop(columns = ['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'result'])
y_test = test[['result']]

#Reggression model
reg_mod = LogisticRegression(max_iter=1000)

reg_mod.fit(x_train, np.ravel(y_train.values))
pred = reg_mod.predict_proba(x_test)
pred = pred[:,1]


#print predictions
for i in range(len(pred)):
    win_prob = np.round(pred[i],2) * 100
    away_team = test.reset_index().drop(columns = 'index').loc[i,'away_abbr']
    home_team = test.reset_index().drop(columns = 'index').loc[i,'home_abbr']
    print(f'{home_team} has a {win_prob}% chance of beating the {away_team}.')
        



xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.05, max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(x_train,y_train)

x_test = agg_test.drop(columns = ['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week','result'])
y_test = agg_test[['result']]
pred = reg_mod.predict_proba(x_test)
pred = pred[:,1]

xgb_preds = xg_reg.predict(x_test)

for i in range(len(pred)):
    win_prob = np.round(pred[i],4) * 100
    away_team = agg_test.reset_index().drop(columns = 'index').loc[i,'away_abbr']
    home_team = agg_test.reset_index().drop(columns = 'index').loc[i,'home_abbr']
    print(f'{home_team} has a {win_prob}% chance of beating the {away_team}.')


for i in range(len(xgb_preds)):
    win_prob = np.round(xgb_preds[i],4) * 100
    away_team = agg_test.reset_index().drop(columns = 'index').loc[i,'away_abbr']
    home_team = agg_test.reset_index().drop(columns = 'index').loc[i,'home_abbr']
    print(f'{home_team} has a {win_prob}% chance of beating the {away_team}.')

        
accuracy = accuracy_score(y_test, np.round(xgb_preds))
