
#%% 

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from config.config_field import GRID_HEIGHT, GRID_WIDTH
from config.config_distance import MINIMUM_SA, MINIMUM_SHOTS 
from utils.states_actions import CODE_ACTIONS, CODE_STATES, \
                                    decodify_states, codify_states
from config.config_Q import GAMMA, LEARNING_RATE, EPOCHS, RANDOMNESS

#%%

#state_action_next = pd.read_csv(os.path.join('data', 'state_action_next.csv'))

san_matrix = pd.read_csv(os.path.join('data', 'san_prob_matrix.csv'),
                         index_col = [0,1])
san_matrix.columns = san_matrix.columns.astype(int)

state_shot_reward = pd.read_csv(os.path.join('data', 'state_shot_reward.csv'))
state_shot_reward.set_index('state', inplace = True)

#%%

F_states = min([k for (x, y, open, pos), k in CODE_STATES.items() if pos])
N_states = len(CODE_STATES)
N_actions = len(CODE_ACTIONS)

vectorized = True

if vectorized:
    
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device')

    Q = torch.zeros(N_states - F_states, N_actions,
                    device = device, dtype = torch.float32)
    R = torch.zeros(N_states - F_states, N_actions,
                    device = device, dtype = torch.float32)
    R[:, 0] = torch.tensor(
        state_shot_reward.sort_index()['reward'].values,
        device = device, dtype = torch.float32)
    P = torch.tensor(
        san_matrix.values,
        device = device, dtype = torch.float32
    ).reshape(N_states - F_states, N_actions, N_states)
    P[:, 0] = 0

    error = torch.zeros(EPOCHS, device = device, dtype = torch.float32)

    for e in tqdm(range(EPOCHS), desc = 'Epochs'):
        
        V, _ = Q.max(dim = 1)
        V = torch.cat([-V, V], dim = 0)
        
        Q_new = R + GAMMA * P @ V
        
        error[e] = (Q_new - Q).abs().mean()
        
        Q += LEARNING_RATE * (Q_new - Q)
        if RANDOMNESS is not None:
            Q += RANDOMNESS * torch.randn_like(Q)

    error = error.cpu().numpy()
    Q = Q.cpu().numpy()

else:

    index_loos = np.arange(F_states)

    Q = pd.DataFrame(0, index = range(F_states, N_states),
                        columns = range(N_actions))

    error = []

    for e in tqdm(range(EPOCHS), desc = 'Epochs'):
        
        Q_old = Q.copy()
        V_old_poss = Q_old.max(axis = 1)
        V_old_loss = - V_old_poss.copy()
        V_old_loss.index = index_loos 
        V_old = pd.concat([V_old_loss, V_old_poss])  
        
        total_error = 0
        
        for state in range(F_states, N_states):
            
            shot_reward = state_shot_reward.loc[state, 'reward']
            
            for action in range(N_actions):
                
                Q_old_sa = Q_old.loc[state, action]
                
                if action == 0:
                    Q_new_sa = shot_reward
                else:
                    prob = san_matrix.loc[state, action]
                    prob.index = prob.index.astype(int)
                    Q_new_sa = GAMMA * prob @ V_old
                
                Q.loc[state, action] = Q_old_sa + LEARNING_RATE * (Q_new_sa - Q_old_sa)
                if RANDOMNESS is not None:
                    Q.loc[state, action] += RANDOMNESS * np.random.randn()
                
                total_error += abs(Q_new_sa - Q_old_sa) / Q.size
                                
        error.append(total_error)

#%%

plt.figure()
plt.plot(error)
plt.yscale("log")
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.savefig(os.path.join('figures', 'error_plot.png'), dpi = 300)

pd.DataFrame(Q, index = range(F_states, N_states)).\
    to_csv(os.path.join('results', 'Q_values.csv'), index = True)

# %%