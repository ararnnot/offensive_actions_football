#%% 

import pandas as pd
import numpy as np
import json

from utils.field_play import draw_state_actions, draw_state_action_values
from utils.states_actions import CODE_STATES, CODE_ACTIONS, \
                                    decodify_states, DECODE_ACTIONS
from utils.tables_latex import table_san, table_values

#%%

matches = pd.read_csv('data/matches_info.csv')
players = pd.read_csv('data/player_names_info.csv')
df = pd.read_csv('data/data_steps_players.csv')

with open('data/dictionaries.json') as f:
    json_dicts = json.load(f)
team_names = json_dicts['team_names']
team_names = {int(k): v for k, v in team_names.items()}
position_map = json_dicts['position_id_mapping']
position_map = {int(k): v for k, v in position_map.items()}
del json_dicts

players['position'] = players['player_position_id'].fillna(-1).astype(int).map(position_map)
players['team'] = players['team_id'].map(team_names)

Q = pd.read_csv('results/Q_values.csv', index_col = 0)
Q.columns = Q.columns.astype(int)

F_states = min([k for (x, y, open, pos), k in CODE_STATES.items() if pos])
N_states = len(CODE_STATES)
N_actions = len(CODE_ACTIONS)

Q_df = Q.reset_index().melt(
    id_vars = 'index', var_name = 'action', value_name = 'quality')
Q_df.rename(columns = {'index': 'state'}, inplace = True)
V_df = Q_df.groupby('state')['quality'].max().reset_index()
V_df.rename(columns = {'quality': 'value'}, inplace = True)

V_df_negative = V_df.copy()
V_df_negative['state'] = V_df_negative['state'] - F_states
V_df_negative['value'] = - V_df_negative['value']
V_df = pd.concat([V_df_negative, V_df], ignore_index = True)
del V_df_negative

#%%


df = df.merge(
    players[['player_id', 'player_name', 'position']].drop_duplicates(),
    on = 'player_id',
    how = 'left'
)
df = df.merge(
    V_df,
    left_on = 'next_state',
    right_on = 'state',
    suffixes = ('', '_next'),
    how = 'left'
)
df = df.drop(columns = 'state_next')
df = df.rename(columns = {'value': 'value_next'})
df = df.merge(
    V_df,
    on = 'state',
    how = 'left'
)
df = df.merge(
    Q_df,
    on = ['state', 'action'],
    how = 'left'
)

df.loc[df['action'] == 0, 'value_next'] = df.loc[df['action'] == 0, 'reward']
df['value_increase'] = df['value_next'] - df['value']
df['value_more_q'] = df['value_next'] - df['quality']

df['x'], df['y'], df['open'], df['pos'] = decodify_states(df['state'])
df['x_next'], df['y_next'], df['open_next'], df['pos_next'] = decodify_states(df['next_state'])
df[['action_type', 'x_act', 'y_act']] = pd.DataFrame(
    df['action'].map(DECODE_ACTIONS).tolist(), index=df.index)

# %%

""" CODE FOOR LOOKING FOR PARITCULAR GOALS
vil_oth = matches.query('(team_A == 221 and team_B == 222) or' + \
                        '(team_A == 222 and team_B == 221)')['match_id'].values


for m in vil_oth:
    print(m)
    events = df.query('match_id == @m')
    events = events.query('player_id == 6766')
    goals = events['reward'].sum()
    print(goals)
match = 3806726
events = df.query('match_id == @match')
idx = events.query('player_id == 6766 and reward == 1').index[0]
print(events.query('player_id == 6766 and reward == 1').index)
"""

# %%

# GOALS
# 1 of https://www.youtube.com/watch?v=cbRVC5vpObs: idx = 584907 (<--)
# 10 of https://www.youtube.com/watch?v=cbRVC5vpObs: idx = 815381
# 1:40 of https://www.youtube.com/watch?v=eWUAldwI6JQ: idx = 816041

idx = 584907
data = df.loc[idx-6:idx]
fig, ax, _, _ = draw_state_actions(
    df = data[['state', 'action', 'next_state', 'reward']],
    add_index = True
)
fig.savefig(f'figures/plays/goal_{idx}.png', dpi = 300)
fig, ax = draw_state_action_values(
    df = data,
    col_text = 'value_increase',
    boxed_background = True,
    txt_size = 12
)
fig.savefig(f'figures/plays/goal_{idx}_val_increase.png', dpi = 300)

# %%


df_print = table_san(data)
df_print.to_latex('results/plays/goal_584907_table.tex', index = False)

df_print = table_values(data)
df_print.to_latex('results/plays/goal_584907_table_val_increase.tex', index = False)


# %%

"""
    Now, table of players
"""

# Now count total value gained by player
players_summary = df.copy().groupby('player_id').agg(
    value_increase_mean = ('value_increase', 'mean'),
    occurrences = ('value_increase', 'count'),
    goals_pm = ('reward', 'sum'),
).reset_index()

players_summary = players_summary.merge(players,
              on = 'player_id',
              how = 'left')
players_summary['goals_pm'] = players_summary['goals_pm'] / players_summary['matches_played']
#players_summary['value_increase_mean'] = players_summary['value_increase_mean'] * players_summary['occurrences'] / players_summary['matches_played']

players_summary = (players_summary    
    .copy()
    .query('matches_played > 10 and occurrences > 500').copy()
    .sort_values('value_increase_mean', ascending = False)
)
    
players_summary[['player_name', 'team', 'position', 'value_increase_mean']] \
    .head(25) \
    .to_latex(f'results/players/table_summary.tex',
              index = False, float_format = "%.4f")

position_tables = {}
for position, group in players_summary.groupby('position'):
    position_tables[position] = group.sort_values('value_increase_mean', ascending = False)
    
print(f'Positions on tables: {position_tables.keys()}')

for position, table in position_tables.items():
    table.insert(0, 'Rank', range(1, len(table) + 1))
    table = table[['Rank', 'player_name', 'team', 'value_increase_mean', 'occurrences', 'goals_pm']]
    table.head(15) \
         .to_latex(f'results/players/table_{position}.tex',
                   index = False, float_format = "%.4f")

# %%
