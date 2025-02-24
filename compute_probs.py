#%% Install dependencies

import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm

from offensive_actions_football.states_actions import CODE_ACTIONS, CODE_STATES, \
                                    sa_out_field, state_flip, \
                                    decodify_states, DECODE_ACTIONS
from offensive_actions_football.lipschitz_extension import Lipschitz_Vector, Lipschitz_Real
from offensive_actions_football.distances import distance_matrix_states, distance_matrix_state_action
from config.config_distance import MINIMUM_SA, MINIMUM_SHOTS, NO_SHOTS_ZERO, \
                                    MAX_PASS_DISTANCE
from config.config_field import SIZE_GRID_FACTOR, GRID_HEIGHT, GRID_WIDTH

#%% Load data

data_path = os.path.join('data', 'data_steps.csv')
data = pd.read_csv(data_path)

#%% State action next state DataFrame

N_states = len({val for (x, y, op, pos), val in CODE_STATES.items()})
F_states = min({val for (x, y, op, pos), val in CODE_STATES.items() if pos == 1})
N_actions = len(CODE_ACTIONS)

state_action = pd.DataFrame(
    [[state, action]
     for state in range(F_states, N_states)
     for action in range(N_actions)],
    columns = ['state', 'action']
)

sa_counts = data.groupby(['state', 'action']) \
                        .size().reset_index(name='sa_occurrences')
state_action = state_action.merge(sa_counts, how = 'left', on = ['state', 'action'])

state_action_next = pd.DataFrame(
    [[state, action, next_state]
     for state in range(F_states, N_states)
     for action in range(N_actions)
     for next_state in range(N_states)],
    columns = ['state', 'action', 'next_state']
)

san_counts = data.groupby(['state', 'action', 'next_state']) \
                .size().reset_index(name='san_occurrences')
state_action_next = (state_action_next
                     .merge(san_counts, how = 'left', on = ['state', 'action', 'next_state'])
                     .fillna(0))
state_action_next = (state_action_next
                     .merge(sa_counts, how = 'left', on = ['state', 'action'])
                     .fillna(0))

# Reward only exists in shot
state_shot_reward = pd.DataFrame(
    [[state] for state in range(F_states, N_states)],
    columns = ['state']
)
state_shot_reward = (state_shot_reward
                     .merge(data.query('action == 0')
                                .groupby('state')
                                .agg(reward_mean=('reward', 'mean'),
                                     occurrences=('reward', 'size'))
                                .reset_index(),
                            how='left',
                            on='state')
                     ).fillna(0)
del sa_counts, san_counts

# %% Probabiliy of next state given state and action

state_action_next['san_prob'] = state_action_next['san_occurrences'] / state_action_next['sa_occurrences']

# Minimum of state and action (or shots) to be considered

state_action_next.loc[state_action_next['sa_occurrences'] < MINIMUM_SA, 'san_prob'] = 0

# Extend the probability function

san_matrix = state_action_next.pivot(
    index = ['state', 'action'],
    columns = 'next_state',
    values = 'san_prob'
)

index_train = san_matrix.index[san_matrix.sum(axis = 1) >= 0.5]
index_test  = san_matrix.index[san_matrix.sum(axis = 1) < 0.5]

train = san_matrix.loc[index_train]
X = np.array(train.index.values.tolist())
Y = train.values

Xnew = np.array(index_test.values.tolist())
Probabilities = Lipschitz_Vector(
    X,
    Y,
    distance_matrix_function = distance_matrix_state_action,
    compute_K = True,
    device = 'available',
    precision = 'single'
)
Ynew = Probabilities.mcshane_whitney_extension(Xnew)

san_matrix.loc[index_test] = Ynew
if (san_matrix < 0).sum().sum() > 0:
    print(f' (!) {(san_matrix < -1e-5).sum().sum()} cells with negative probability.')
if (san_matrix.sum(axis = 1) < 0.2).sum() > 0:
    print(f' (!) {(san_matrix.sum(axis = 1) < 0.2).sum()} rows with less than 0.2 prob.')
san_matrix = san_matrix.div(san_matrix.sum(axis=1), axis=0)

# Some carries directed out of the game -> loos possession
sa_out = sa_out_field()
for state, action in sa_out.query('out == True')[['state', 'action']].values:
    san_matrix.loc[(state, action)] = 0
    san_matrix.loc[(state, action), state_flip(state, open_to = False)] = 1
    
# Avoid carries from not oepn play
for state in range(F_states, N_states):
    for action in range(N_actions):
        x, y, op, pos = decodify_states(state, copy = False)
        act, ax, ay = DECODE_ACTIONS[action]
        if not op and act == 43:
            san_matrix.loc[(state, action)] = 0
            san_matrix.loc[(state, action), state_flip(state, open_to = False)] = 1
            
# Penalty probability very long passes or shots
if MAX_PASS_DISTANCE is not None:
    for state in range(F_states, N_states):
        for action in range(N_actions):
            x, y, op, pos = decodify_states(state, copy = False)
            act, ax, ay = DECODE_ACTIONS[action]
            if act in [16, 30]:
                if act == 16: ax, ay = GRID_WIDTH - 1, GRID_HEIGHT // 2
                dist = ((x - ax) ** 2 + (y - ay) ** 2) ** 0.5 / SIZE_GRID_FACTOR
                if (op) and (dist > MAX_PASS_DISTANCE[0]) or \
                    (not op) and (dist > MAX_PASS_DISTANCE[1]):
                    san_matrix.loc[(state, action)] = 0
                    state_dir = CODE_STATES[(ax, ay, True, True)]
                    state_dir = state_flip(state_dir, open_to = True)
                    san_matrix.loc[(state, action), state_dir] = 1

state_action_next.rename(columns = {'san_prob': 'san_prob_old'}, inplace = True)
state_action_next = pd.merge(
    state_action_next,
    san_matrix.reset_index().melt(id_vars = ['state', 'action'],
                                  var_name='next_state', value_name='san_prob'),
    how='left',
    on=['state', 'action', 'next_state']
)

# %% Shot goal probability function

# Minimum shots to be considered
index_train = state_shot_reward.query('occurrences >= @MINIMUM_SHOTS').index
index_test  = state_shot_reward.query('occurrences <  @MINIMUM_SHOTS').index

if NO_SHOTS_ZERO:
    
    Ynew = 0
    
else:
    
    train = state_shot_reward.loc[index_train]
    X = train['state'].values
    Y = train['reward_mean'].values

    Goal_prob = Lipschitz_Real(
        X,
        Y,
        distance_matrix_function = distance_matrix_states,
        compute_K = True,
        device = 'available',
        precision = 'single'
    )
    Xnew = state_shot_reward.loc[index_test, 'state'].values
    Ynew = Goal_prob.mcshane_whitney_extension(Xnew)

state_shot_reward.loc[index_train, 'reward'] = state_shot_reward.loc[index_train, 'reward_mean']
state_shot_reward.loc[index_test, 'reward'] = Ynew

#%% SAVE DATA

san_matrix.to_csv(os.path.join('data', 'san_prob_matrix.csv'), index = True)
state_shot_reward.to_csv(os.path.join('data', 'state_shot_reward.csv'), index = False)
state_action_next.to_csv(os.path.join('data', 'state_action_next.csv'), index = False)


# %%
