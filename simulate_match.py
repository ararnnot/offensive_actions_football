
#%%

import pandas as pd
import numpy as np
import os
import io
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

from config.config_field import GRID_HEIGHT, GRID_WIDTH
from utils.states_actions import decodify_states, CODE_STATES
from utils.field_play import draw_state_actions

#%%

san_matrix = pd.read_csv(os.path.join('data', 'san_prob_matrix.csv'), index_col = [0,1])
state_shot_reward = pd.read_csv(os.path.join('data', 'state_shot_reward.csv'))
Q = pd.read_csv(os.path.join('results', 'Q_values.csv'), index_col = 0)

san_matrix.columns = san_matrix.columns.astype(int)
Q.columns = Q.columns.astype(int)
state_shot_reward.index = state_shot_reward['state']
V = Q.max(axis = 1)

np.random.seed(42)

state = 292
N_events = 500

CHOOSE_ACTION = 'ranked_exp'

#%%

events = pd.DataFrame(columns=['state', 'action', 'next_state', 'reward', 'value', 'team_A'])
team_A = True

for i in tqdm(range(N_events), desc = 'Simulating match'):
    
    if CHOOSE_ACTION == 'max':
        action = Q.loc[state].idxmax()
    elif CHOOSE_ACTION == 'random':
        action = np.random.choice(Q.columns)
    elif CHOOSE_ACTION == 'softmax':
        softmax = np.exp(Q.loc[state]) / np.exp(Q.loc[state]).sum()
        action = np.random.choice(Q.columns, p = softmax)
    elif CHOOSE_ACTION == 'softmax_pos':
        row = Q.loc[state]
        row -= row.max()
        row *= 5
        softmax = np.exp(row) / np.exp(row).sum()
        action = np.random.choice(Q.columns, p = softmax)
    elif CHOOSE_ACTION == 'ranked_exp':
        sorted_actions = Q.loc[state].copy().sort_values(ascending=False)
        ranks = np.arange(1, len(sorted_actions) + 1)
        probabilities = 0.75 ** (ranks - 1)
        probabilities /= probabilities.sum()
        action = np.random.choice(sorted_actions.index, p = probabilities)

        
    reward = 0
    
    # The case of shoot is diferent    
    if action == 0:
        reward_prob = state_shot_reward.loc[state, 'reward']
        reward = 1 if np.random.rand() < reward_prob else 0
        if reward == 1:
            next_state = CODE_STATES.get((GRID_WIDTH // 2, GRID_HEIGHT // 2,
                                            False, False))
        else:
            y = int(GRID_HEIGHT // 2 + np.random.choice([-1, 0, 1]))
            next_state = CODE_STATES.get((0, y, False, False))
    else:
        next_state = np.random.choice(
            san_matrix.loc[(state, action)].index,
            p = san_matrix.loc[state, action]
        )
    
    value = V[state]
    if not team_A: value = - value
    events.loc[i] = [state, action, next_state, reward, value, team_A]
    
    x, y, op, pos = decodify_states(next_state, copy = False)
    if not pos:
        team_A = not team_A
        next_state = CODE_STATES[(x, y, op, True)]
    state = next_state


#%% Video

frames = []
team_A = True
scoreboard = [0, 0]

for i in tqdm(range(N_events), desc = 'Creating gif'):
    
    if events.loc[i, 'reward'] == 1:
        scoreboard[int(not team_A)] += 1
        
    start = i - 8 if i - 8 >= 0 else 0
    fig, ax, team_A, open_play_n = draw_state_actions(
        events[start:i], alpha_increase = True, team_A = team_A,
        init = i, scoreboard = scoreboard, value_col = 'value')
    if i - 8 < 0 and not team_A: team_A = not team_A
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    
    buf.seek(0)
    frame = Image.open(buf)
    frames.append(frame)
    if not open_play_n: frames.append(frame)

gif_path = "figures/maches/match_simulated.gif"
frames[0].save(gif_path, save_all = True, append_images = frames[1:],
               duration = 400, loop = 0)
print(f'Gif saved in {gif_path}')