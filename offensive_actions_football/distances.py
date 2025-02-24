
import pandas as pd
import numpy as np
import torch

from tqdm import tqdm

from config.config_field import FIELD_HEIGHT, FIELD_WIDTH, GRID_HEIGHT, GRID_WIDTH
from offensive_actions_football.states_actions import DECODE_ACTIONS, decodify_states
from offensive_actions_football.field import get_centers_torch

from config.config_distance import OPEN_PLAY_PEN, DIFERENT_ACTION_PEN, \
                                    SYMMETRIC_FACTOR, INCREMENT_FACTOR, \
                                    INITIAL_FACTOR, DIRECTED_FACTOR, \
                                    LENGTH_FACTOR

def dtype_string(dtype):
    if dtype == 'half': return torch.float16
    if dtype == 'float': return torch.float32
    if dtype == 'double': return torch.float64
    return torch.float32

def distance_points(x1, y1, x2, y2,
                    min_factor = 0.8, sector_xy = True,
                    dtype = torch.float32, device = 'cpu'):
    """
    Compute the euclidean-modified distance between two points.
    
    Parameters
        x1, y1, x2, y2 : (int, vector or torch.Tensor)
        min_factor : (float) simmetric factor for the y-axis.
    Returns
        torch.Tensor : distance between each two points.
    """
    
    if not isinstance(x1, torch.Tensor):
        x1 = torch.tensor(x1, dtype = dtype, device = device)
    if not isinstance(y1, torch.Tensor):
        y1 = torch.tensor(y1, dtype = dtype, device = device)
    if not isinstance(x2, torch.Tensor):
        x2 = torch.tensor(x2, dtype = dtype, device = device)
    if not isinstance(y2, torch.Tensor):
        y2 = torch.tensor(y2, dtype = dtype, device = device)
    
    if sector_xy:
        x1, y1 = get_centers_torch(x1, y1)
        x2, y2 = get_centers_torch(x2, y2)
    
    dif2_x = ( x1 - x2 ) ** 2
    dif2_y = ( y1 - y2 ) ** 2
    
    flip_y = ( y1 - (FIELD_HEIGHT - y2) ).abs()
    min2_y  = torch.min( dif2_y, flip_y )
    
    dist = (
            dif2_x +
            dif2_y * (1-min_factor) +
            min2_y * min_factor
         ) ** 0.5
    
    return dist, x1, y1, x2, y2

def distance_states(state1, state2,
                    dtype = torch.float32, device = 'cpu'):
    """
    Compute a defined distance between two states.
    
    Parameters
        state1, state2 : (int, vector or torch.Tensor)
    Returns
        torch.Tensor: distance between each two states.
    """
    
    if not isinstance(state1, torch.Tensor):
        state1 = torch.tensor(state1, dtype = dtype, device = device)
    if not isinstance(state2, torch.Tensor):
        state2 = torch.tensor(state2, dtype = dtype, device = device)
    
    x1, y1, open1, poss1 = decodify_states(state1.clone(),
                                           copy = False)
    x2, y2, open2, poss2 = decodify_states(state2.clone(),
                                           copy = False)
    
    dist_0, x1, y1, x2, y2 = distance_points(
        x1, y1, x2, y2, min_factor = SYMMETRIC_FACTOR,
        dtype = dtype, device = device)
        
    dist = dist_0 + OPEN_PLAY_PEN * (open1 != open2)
    
    return dist, x1, y1, x2, y2

def distance_matrix_states(states1, states2, show_progress = True,
                           dtype = torch.float32, device = 'cpu'):    
    
    if not isinstance(states1, torch.Tensor):
        states1 = torch.tensor(states1, dtype = dtype, device = device)
    if not isinstance(states2, torch.Tensor):
        states2 = torch.tensor(states2, dtype = dtype, device = device)
    
    distance_matrix = torch.zeros(len(states1), len(states2),
                                  dtype = dtype, device = device)
    for i in tqdm(range(len(states1)),
                  desc = 'Computing distance matrix os states',
                  disable = not show_progress):
        col, _, _, _, _ = distance_states(
                            torch.full((len(states2),), states1[i],
                                       dtype = dtype, device = device),
                            states2,
                            dtype = dtype, device = device)
        distance_matrix[i,:] = col
        
    return distance_matrix

def dif_length(x1, y1, x2, y2):
    
    l1 = (x1 ** 2 + y1 ** 2) ** 0.5
    l2 = (x2 ** 2 + y2 ** 2) ** 0.5
    return (l1 - l2).abs()

def get_action_information(action,
                           dtype = torch.float32, device = 'cpu'):
    """
    Gets the action, x and y coordinates of an action.
    
    Parameters
        action : (int, vector or torch.Tensor)
    Returns
        torch.tensors (1D): action (event id), x and y coordinates
    """
    
    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action, dtype = dtype, device = device)

    decoded_actions = [DECODE_ACTIONS.get(a) for a in action.flatten().tolist()]
    act, x, y = zip(*decoded_actions)
    
    act = torch.tensor(act, dtype = dtype, device = device)
    x = torch.tensor(x, dtype = dtype, device = device)
    y = torch.tensor(y, dtype = dtype, device = device)
    
    x, y = get_centers_torch(x, y)

    return act, x, y

def distance_state_action(state1, action1, state2, action2,
                          dtype = torch.float32, device = 'cpu'):
    
    # Distance between states
    dist, x1, y1, x2, y2 = distance_states(state1, state2,
                                           dtype = dtype, device = device)
    dist = dist * INITIAL_FACTOR
    
    act1, xd1, yd1 = get_action_information(action1,
                                            dtype = dtype, device = device)
    act2, xd2, yd2 = get_action_information(action2,
                                            dtype = dtype, device = device)
    
    xd1[act1 == 43] = x1[act1 == 43] + xd1[act1 == 43] - FIELD_WIDTH / (2 * GRID_WIDTH)
    yd1[act1 == 43] = y1[act1 == 43] + yd1[act1 == 43] - FIELD_HEIGHT / (2 * GRID_HEIGHT)
    xd1[act1 == 16], yd1[act1 == 16] = FIELD_WIDTH, FIELD_HEIGHT / 2
    
    
    # Distance betwen increment on x and y
    dist_, _, _, _, _ = distance_points(
        x1 - xd1, y1 - yd1, x2 - xd2, y2 - yd2,
        min_factor = SYMMETRIC_FACTOR, sector_xy = False,
        dtype = dtype, device = device)
    dist += dist_ * INCREMENT_FACTOR
        
    # Distance between directed action
    dist_, _, _, _, _ = distance_points(
        xd1, yd1, xd2, yd2,
        min_factor = 0, sector_xy = False,
        dtype = dtype, device = device)
    dist += dist_ * DIRECTED_FACTOR
    
    # Legth difference
    dist += dif_length(xd1, yd1, xd2, yd2) * LENGTH_FACTOR

    dist[act1 != act2] += DIFERENT_ACTION_PEN
    
    return dist
    
def distance_matrix_state_action(comb1, comb2, show_progress = True,
                                 dtype = torch.float32, device = 'cpu',
                                 vectorized = False):
    """
    Compute the distance matrix between two state-action combinations.
    Parameters
        comb1, comb2 : (torch.Tensor 2-dimensional where
            the first column is the state and the second the action)
    Returns
        torch.Tensor: distance between each two states.
    """
    
    states1 = comb1[:,0].clone()
    actions1 = comb1[:,1].clone()
    states2 = comb2[:,0].clone()
    actions2 = comb2[:,1].clone()
    
    if not vectorized:
        distance_matrix = torch.zeros(len(states1), len(states2),
                                    dtype = dtype, device = device)
        for i in tqdm(range(len(states1)),
                    desc = 'Computing distance matrix of state-action',
                    disable = not show_progress):
            distance_matrix[i] = distance_state_action(
                torch.full((len(states2),), states1[i],
                        dtype = dtype, device = device),
                torch.full((len(actions2),), actions1[i],
                        dtype = dtype, device = device),
                states2,
                actions2,
                dtype = dtype, device = device)
    else:
        grid_s1, grid_s2 = torch.meshgrid(states1, states2, indexing='ij')
        grid_a1, grid_a2 = torch.meshgrid(actions1, actions2, indexing='ij')
        grid_s1 = grid_s1.flatten()
        grid_a1 = grid_a1.flatten()
        grid_s2 = grid_s2.flatten()
        grid_a2 = grid_a2.flatten()        
        distance_matrix = distance_state_action(grid_s1, grid_a1, grid_s2, grid_a2,
                                                dtype = dtype, device = device)
        distance_matrix = distance_matrix.reshape(len(states1), len(states2))
        
    return distance_matrix  
    
    
    
    
    
    
