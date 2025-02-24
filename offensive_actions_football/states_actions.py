# Codifies the states and actions

import config.config_field as config_field

import torch
import numpy as np
import pandas as pd
from itertools import product as iterproduct
 
CODE_STATES = {
    (x, y, open_play, possession): idx
    for idx, (possession, open_play, y, x) in enumerate(
        iterproduct(
            range(2),
            range(2),
            range(config_field.GRID_HEIGHT),
            range(config_field.GRID_WIDTH)
        )
    )
}

DECODE_STATES = {
    idx: (x, y, open_play, possession)
    for (x, y, open_play, possession), idx in CODE_STATES.items()
}

def to_int(data):
    if isinstance(data, (pd.DataFrame, pd.Series)): return data.astype(int)
    elif isinstance(data, np.ndarray): return data.astype(int)
    elif isinstance(data, torch.Tensor): return data.long()
    return data

def codify_states(x, y, open_play, possesion):
    
    if isinstance(x, list): x = np.array(x)
    if isinstance(y, list): y = np.array(y)
    if isinstance(open_play, list): open_play = np.array(open_play)
    if isinstance(possesion, list): possesion = np.array(possesion)
    
    open_play = to_int(open_play)
    possesion = to_int(possesion)
    
    codif = (
        possesion * config_field.GRID_WIDTH * config_field.GRID_HEIGHT * 2 +
        open_play * config_field.GRID_WIDTH * config_field.GRID_HEIGHT +
        y * config_field.GRID_WIDTH +
        x
    )
    
    return codif

def decodify_states(codif, copy = True):
    
    if copy:
        codif = codif.copy()
    
    x = codif % config_field.GRID_WIDTH
    codif //= config_field.GRID_WIDTH
    
    y = codif % config_field.GRID_HEIGHT
    codif //= config_field.GRID_HEIGHT
    
    open_play = codif % 2
    codif //= 2
    
    possesion = codif
    
    return x, y, open_play, possesion

# CODE_ACTIONS
# shot: 16, carry: 43 + move x-y, pass: 30 + directed x-y

CODE_ACTIONS = {
    (16, 0, 0): 0 
}

CODE_ACTIONS.update({
    (43, x, y): idx + len(CODE_ACTIONS)
    for (x, y), idx in config_field.CARRY_CODIFICATION.items()
})

CODE_ACTIONS.update({
    (30, x, y): idx + len(CODE_ACTIONS)
    for idx, (y, x) in enumerate(
        iterproduct(
            range(config_field.GRID_HEIGHT),
            range(config_field.GRID_WIDTH)
        )
    )
})

DECODE_ACTIONS = {
    idx: (event_type, x, y)
    for (event_type, x, y), idx in CODE_ACTIONS.items()
}   

def codify_actions(event_type, x, y, directed_x, directed_y):
    
    action = pd.Series([None] * len(event_type), index=event_type.index)
    
    action[event_type == 16] = 0
    for i, (dx, dy, _x, _y) in enumerate(zip(directed_x, directed_y, x, y)):
        if event_type.iloc[i] == 43:
            key = (dx - _x, dy - _y)
            action.iloc[i] = config_field.CARRY_CODIFICATION[key] + 1
    action[event_type == 30] = directed_y * config_field.GRID_WIDTH + \
            directed_x + len(config_field.CARRY_CODIFICATION) + 1
    
    return action

def states_combined(x, y, open_play, possesion):
    """
    Returns the codification of all possible combined states
    """
    
    combinations = pd.DataFrame(
        list(iterproduct(x, y, open_play, possesion)),
        columns=['x', 'y', 'open_play', 'possession']
    )
    states = codify_states(
        combinations['x'], combinations['y'],
        combinations['open_play'], combinations['possession']
    )
    return states.values
    
def sa_out_field():
    
    N_states = len(CODE_STATES)
    F_states = min([k for (x, y, open_play, pos), k in CODE_STATES.items() if pos])
    N_actions = len(CODE_ACTIONS)
    
    sa_out = pd.DataFrame(
        [[state, action, False]
         for state in range(F_states, N_states)
         for action in range(N_actions)],
        columns = ['state', 'action', 'out']
    )
    
    states = states_combined([0],
                             range(config_field.GRID_HEIGHT),
                             range(2), [True])
    actions = [CODE_ACTIONS.get((43, -1, y)) for y in [-1,0,1]]
    sa_out.loc[sa_out['state'].isin(states) & sa_out['action'].isin(actions), 'out'] = True
    
    states = states_combined([config_field.GRID_WIDTH - 1],
                             range(config_field.GRID_HEIGHT),
                             range(2), [True])
    actions = [CODE_ACTIONS.get((43, 1, y)) for y in [-1,0,1]]
    sa_out.loc[sa_out['state'].isin(states) & sa_out['action'].isin(actions), 'out'] = True
    
    states = states_combined(range(config_field.GRID_WIDTH),
                                [0],
                                range(2), [True])
    actions = [CODE_ACTIONS.get((43, x, -1)) for x in [-1,0,1]]
    sa_out.loc[sa_out['state'].isin(states) & sa_out['action'].isin(actions), 'out'] = True
    
    states = states_combined(range(config_field.GRID_WIDTH),
                                [config_field.GRID_HEIGHT - 1],
                                range(2), [True])
    actions = [CODE_ACTIONS.get((43, x, 1)) for x in [-1,0,1]]
    sa_out.loc[sa_out['state'].isin(states) & sa_out['action'].isin(actions), 'out'] = True
        
    return sa_out

def state_flip(state, open_to = None):
    """
    Flips the state to the other side of the field
    """
    
    x, y, open_play, possession = decodify_states(state, copy = False)
    x = config_field.GRID_WIDTH - 1 - x
    y = config_field.GRID_HEIGHT - 1 - y
    possession = 1 - possession
    if open_to is not None:
        open_play = open_to
        
    return codify_states(x, y, open_play, possession)
