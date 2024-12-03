#%% Install dependencies
 
import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm

from utils.field import convert_field_grid, carry_line
from utils.states_actions import codify_states, codify_actions

#%% Import data

csv_dir = 'data/all_matches'

df_list = []

for root, dirs, files in os.walk(csv_dir):
    for file_name in files:
        if file_name.endswith('.csv'):
            file_path = os.path.join(root, file_name)
            df = pd.read_csv(file_path)
            df_list.append(df)

df = pd.concat(df_list, ignore_index=True)
del df_list

print(f'Data imported. {len(df)} events')

#%% Extract names

# the sencod type is more specific
event_types = (
    df.groupby(['event_type_id', 'event_type_name', 'type_id', 'type_name', 'outcome_id', 'outcome_name'],
               dropna = False)
    .size()
    .reset_index(name = 'count')
    .sort_values('event_type_id')
    .reset_index(drop = True)
)

event_names = dict(
    df[['event_type_id', 'event_type_name']]
    .drop_duplicates()
    .reset_index(drop = True)
    .values
)

event_sub_names = dict(
    df[['type_id', 'type_name']]
    .drop_duplicates()
    .reset_index(drop = True)
    .values
)

outcome_names = dict(
    df[['outcome_id', 'outcome_name']]
    .drop_duplicates()
    .reset_index(drop = True)
    .values
)

play_pattern_names = dict(
    df[['play_pattern_id', 'play_pattern_name']]
    .drop_duplicates()
    .reset_index(drop = True)
    .values
)

# Team, players and matches

team_names = dict(
    df[['team_id', 'team_name']]
    .drop_duplicates()
    .reset_index(drop = True)
    .values
)

player_position = dict(
    df[['player_position_id', 'player_position_name']]
    .drop_duplicates()
    .reset_index(drop = True)
    .values
)               

player_info_pos = (
    df.groupby(['player_id', 'player_position_id'])
      .size()
      .reset_index(name = 'count')
      .sort_values(['player_id', 'count'],
                   ascending = [True, False])
      .drop_duplicates(subset = ['player_id'])
      .drop(columns = 'count')
)

player_names = (
    df[['player_id', 'player_name', 'team_id', 'match_id']]
    .drop_duplicates()
    .reset_index(drop=True)
).merge(
    player_info_pos, on = 'player_id', how = 'left'
)

del player_info_pos

match_counts = (
    player_names.groupby(['player_id', 'team_id'])
    .match_id.nunique()
    .reset_index(name = 'matches_played')
)

player_names = (
    player_names
    .drop(columns=['match_id'])
    .drop_duplicates()
    .merge(match_counts, on = ['player_id', 'team_id'], how = 'left')
)
player_names['player_id'] = player_names['player_id'].fillna(-1).astype(int)

del match_counts

matches_info = (
    df[['match_id', 'team_id']]
    .drop_duplicates()
    .groupby('match_id')['team_id']
    .apply(list)
    .reset_index()
)
matches_info[['team_A', 'team_B']] = (
    pd.DataFrame(
        matches_info['team_id'].tolist(),
        index = matches_info.index)
)
matches_info = matches_info.drop(columns = 'team_id')                   

position_id_mapping = {
    1: 'Goalkeeper',
    2: 'Defender',       # Right Back
    3: 'Defender',       # Right Center Back
    4: 'Defender',       # Center Back
    5: 'Defender',       # Left Center Back
    6: 'Defender',       # Left Back
    7: 'Defender',       # Right Wing Back
    8: 'Defender',       # Left Wing Back
    9: 'Midfielder',     # Right Defensive Midfield
    10: 'Midfielder',    # Center Defensive Midfield
    11: 'Midfielder',    # Left Defensive Midfield
    12: 'Midfielder',    # Right Midfield
    13: 'Midfielder',    # Right Center Midfield
    14: 'Midfielder',    # Center Midfield
    15: 'Midfielder',    # Left Center Midfield
    16: 'Midfielder',    # Left Midfield
    17: 'Forward',       # Right Wing
    18: 'Midfielder',    # Right Attacking Midfield
    19: 'Midfielder',    # Center Attacking Midfield
    20: 'Midfielder',    # Left Attacking Midfield
    21: 'Forward',       # Left Wing
    22: 'Forward',       # Right Center Forward
    23: 'Forward',       # Center Forward
    24: 'Forward',       # Left Center Forward
}           
                    
# %% Extract columns

# Drop duplicates: shots with events giving extra information
df = df.drop_duplicates(subset = ['index', 'match_id'], keep = 'first')
df.to_csv('data/all_events_matches.csv')
print(f'Data saved. {len(df)} events')

def time_to_float(str):
    if str == 'nan':
        return None
    split = str.split(':')
    if len(split) == 2:
        minutes, seconds = split
    else:
        _, minutes, seconds = str.split(':')
    seconds = float(seconds)
    return 60 * int(minutes) + seconds

df['time'] = df['timestamp'].apply(time_to_float)

df = df[[
    'match_id', 'period', 'time',
    'team_id', 'player_id',
    'event_type_id', 'type_id',
    'location_x', 'location_y',
    'end_location_x', 'end_location_y',
    'outcome_id'
]]

#%% Filter

# Filter: (dribbles 14), shots, passes, and carries
# Dribbles are alwais after a carry (98%) and all information is given by before-after event
df = df.query('event_type_id in [16, 30, 43]')

convert_field_grid(df, 'location_x', 'location_y')
convert_field_grid(df, 'end_location_x', 'end_location_y')

df = df.rename(columns = {
    'location_x': 'sector_x',
    'location_y': 'sector_y',
    'end_location_x': 'sector_directed_x',
    'end_location_y': 'sector_directed_y'
})

df.reset_index(drop = True, inplace = True)

print(f'Data filtered. {len(df)} events')

#%% Run all events to modify carries and other

# Carry: convert to many moves
new_df = []
for index, row in tqdm(df.iterrows()):
    
    if row['event_type_id'] == 43:
        
        path = carry_line(
            row['sector_x'], row['sector_y'],
            row['sector_directed_x'], row['sector_directed_y'],
        )
        
        # Skyp no move carries
        if len(path) > 1:
            for start, end in zip(path[:-1], path[1:]):
                new_row = row.copy()
                new_row['sector_x'], new_row['sector_y'] = start
                new_row['sector_directed_x'], new_row['sector_directed_y'] = end
                new_df.append(new_row)
    
    else:
        new_df.append(row)

df = pd.DataFrame(new_df).reset_index(drop = True)

# Open play (vs set piece): Open Play, Recovery, Interception, NaN
df['open_play'] = df['type_id'].isin([87, 66, 64])
df.loc[df['type_id'].isna(), 'open_play'] = True

# Goal or not
df['goal'] = (df['event_type_id'] == 16) & (df['outcome_id'] == 97)

print(f'Data converted. {len(df)} events')

#%% From df to state, action, new_state, reward

state_df = df[['sector_x', 'sector_y', 'open_play']].copy()
state_df['possession'] = True

states = codify_states(
    state_df['sector_x'], state_df['sector_y'],
    state_df['open_play'], state_df['possession']
)

actions_df = df[['event_type_id', 'sector_x', 'sector_y', 'sector_directed_x', 'sector_directed_y']].copy()
actions = codify_actions(
    actions_df['event_type_id'],
    actions_df['sector_x'], actions_df['sector_y'],
    actions_df['sector_directed_x'], actions_df['sector_directed_y']
)

state_df['possession'] = df['team_id'] == df['team_id'].shift(+1)

states_next = codify_states(
    state_df['sector_x'], state_df['sector_y'],
    state_df['open_play'], state_df['possession']
).shift(-1).fillna(-1).astype(int)

rewards = df['goal'].astype(int)

df_steps = pd.DataFrame({
    'match_id': df['match_id'].astype(int),
    'team_id': df['team_id'].astype(int),
    'player_id': df['player_id'].astype(int),
    'period': df['period'].astype(int),
    'time': df['time'],
    'state': states,
    'action': actions,
    'next_state': states_next,
    'reward': rewards
})

df_steps['next_team_id'] = df_steps['team_id'].shift(-1).fillna(-1).astype(int)
df_steps['next_player_id'] = df_steps['player_id'].shift(-1).fillna(-1).astype(int)

remove_indices = df.loc[
    (df['match_id'] != df['match_id'].shift(-1)) |
    (df['period'] != df['period'].shift(-1))
].index

df_steps = df_steps.drop(remove_indices).reset_index(drop = True)

#%% Save data

df_steps.to_csv('data/data_steps_players.csv', index = False)
df_steps.drop(columns = ['match_id', 'team_id', 'player_id',
                         'period', 'time',
                         'next_team_id', 'next_player_id']) \
    .to_csv('data/data_steps.csv', index = False)

all_dicts = {
    'team_names': team_names,
    'event_names': event_names,
    'event_sub_names': event_sub_names,
    'outcome_names': outcome_names,
    'play_pattern_names': play_pattern_names,
    'player_position': player_position,
    'position_id_mapping': position_id_mapping
}

with open('data/dictionaries.json', 'w') as f:
    json.dump(all_dicts, f, indent = 4)
    
event_types.to_csv('data/event_types_info.csv', index = False)
player_names.to_csv('data/player_names_info.csv', index = False)
matches_info.to_csv('data/matches_info.csv', index = False)
    
print('Data saved')
    
# %%
