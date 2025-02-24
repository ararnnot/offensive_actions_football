
#%% 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from offensive_actions_football.field_plots import draw_values, draw_action, \
                                draw_pitch, draw_separators  
from offensive_actions_football.states_actions import codify_states, decodify_states, \
                                    CODE_ACTIONS, CODE_STATES, DECODE_ACTIONS
from config.config_distance import MINIMUM_SA, MINIMUM_SHOTS 
from config.config_field import GRID_WIDTH, GRID_HEIGHT

#%%

"""
    EVENTS STATISTICS
"""

df = pd.read_csv('data/all_events_matches.csv')

print(f'Number of rows: {df.shape[0]}')
print(f'Number of columns: {df.shape[1]}')
print(f'Number of matches: {df.match_id.nunique()}')


fig, ax = draw_pitch(color = 'white', background = 'green')
draw_separators(ax = ax, alpha = 0.4)
plt.savefig('figures/data/pitch_grid.png', bbox_inches = 'tight', dpi = 300)


df['event_type_name'] = df['event_type_name'].replace({'Ball Receipt*': 'Ball Receipt'})
event_type_counts = df['event_type_name'].value_counts()
print(event_type_counts)
all_types = [
        'Pass','Ball Receipt', 'Carries', 'Pressure', 'Ball Recovery', 'Duel',
       'Clearance', 'Block', 'Foul Committed', 'Dribble', 'Foul Won',
       'Goal Keeper', 'Miscontrol', 'Shot', 'Dispossessed', 'Interception',
       'Dribbled Past', 'Substitution', 'Injury Stoppage', 'Half Start',
       'Half End', 'Tactical Shift', 'Starting XI', '50/50',
       'Referee Ball-Drop', 'Shield', 'Bad Behaviour', 'Player Off',
       'Player On', 'Error', 'Offside', 'Own Goal Against', 'Own Goal For']

plot_types = [
        'Pass','Ball Receipt', 'Carries', 'Pressure', 'Ball Recovery', 'Duel',
       'Clearance', 'Block', 'Foul Committed', 'Dribble', 'Foul Won',
       'Goal Keeper', 'Miscontrol', 'Shot', 'Dispossessed', 'Interception',
       'Dribbled Past', '50/50', 'Referee Ball-Drop', 'Shield',
       'Offside', 'Own Goal Against', 'Own Goal For']

unsued_types = [t for t in all_types if t not in plot_types]

non_used = event_type_counts.loc[unsued_types].sum()
print(f'Non used types: {non_used / event_type_counts.sum()}')

event_type_counts_plot = event_type_counts.reindex(plot_types).fillna(0)

fig, ax = plt.subplots(figsize = (8, 5))

plt.bar(
    event_type_counts_plot.index, 
    event_type_counts_plot.values,
    width=0.7
)

plt.xticks(rotation = 90, fontsize = 14)
plt.yticks(fontsize = 14)

ax.get_yaxis().set_major_formatter(mtick.ScalarFormatter())
ax.ticklabel_format(style = 'plain', axis = 'y')

plt.savefig('figures/data/events_num.png', bbox_inches = 'tight', dpi = 300)
plt.show()


#%%

"""
    IMAGES OF STATE - ACTION - NEXT STATE - PROBABILITY
"""

state_action_next = pd.read_csv(os.path.join('data', 'state_action_next.csv'))
san_matrix = pd.read_csv(os.path.join('data', 'san_prob_matrix.csv'),
                         index_col = [0,1])
state_shot_reward = pd.read_csv(os.path.join('data', 'state_shot_reward.csv'))


#%%

### IMAGES OF REWARD
print('Preparing images reward.')

state_shot_reward['real'] = state_shot_reward['occurrences'] >= MINIMUM_SHOTS

fig, ax = draw_values(
    state_shot_reward,
    'state',
    'reward',
    col_highlight = 'real',
    possession = True,
    open_play = True,
    interval = [None, 1],
    rotate_text = True,
    fontsize = 13
)
fig.savefig('figures/results/reward_openplay.png', bbox_inches = 'tight', dpi = 300)
plt.close()

fig, ax = draw_values(
    state_shot_reward,
    'state',
    'reward',
    col_highlight = 'real',
    possession = True,
    open_play = False,
    interval = [None, 1],
    rotate_text = True
)
fig.savefig('figures/results/reward_setpiece.png', bbox_inches = 'tight', dpi = 300)


#%%

"""
    IMAGES OF SPECIFIC SITUATIONS
"""

### IMAGES OF SPECIFIC SITUATION
print('Preparing images of speceific situation.')

state_to_see = 467
state_to_see = 336
action_to_see = 85

draw_values(
    state_action_next.query('state == @state_to_see and action == @action_to_see'),
    'next_state',
    'san_prob',
    state_highlight_all = state_to_see,
    action_highlight_all = action_to_see,
    possession = True,
    open_play = True,
    interval = [None, 1],
    boxed_highlight = False
)

# %%

# Exisitinvs vs extended actions (corner example)
print('Preparing images of exisitinvs vs extended actions (corner example).')

x = 12; y = 8; open = False; poss = True
state_to_see = codify_states(x, y, open, poss)
print(f'State: ({x}, {y}, {open}, {poss}): {state_to_see}')

act = 30; xa = 12; ya = 2
action_to_see = CODE_ACTIONS.get((act, xa, ya))
print(f'Action: ({act}, {xa}, {ya}): {action_to_see}')

state_action = state_action_next[['state', 'action', 'sa_occurrences']].\
                copy().drop_duplicates()

state_action['real'] = state_action['sa_occurrences'] >= MINIMUM_SA
state_action['real_color'] = state_action['real']. \
                                    apply(lambda x: 'blue' if x else 'red')

fig, _ = draw_action(
    state_action[state_action['real']].query('state == @state_to_see'),
    'state',
    'action',
    col_color = 'real_color',
    possession = poss,
    open_play = open
)

fig.savefig('figures/probabilities/' + \
            f'state_{state_to_see}_real_actions.png',
            bbox_inches = 'tight', dpi = 300)
plt.close()

for p in [True, False]:
    for o in [True, False]:
        
        fig, _ = draw_values(
            state_action_next.query('state == @state_to_see and action == @action_to_see'),
            'next_state',
            'san_prob',
            state_highlight_all = state_to_see,
            action_highlight_all = action_to_see,
            possession = p,
            open_play = o,
            interval = [None, 1],
            boxed_highlight = False,
            fontsize = 15,
            rotate_text = True,
            rotate_legend = True
        )
        
        fig.savefig('figures/probabilities/' + \
                    f'state_{state_to_see}_action_{action_to_see}_poss_{p}_open_{o}.png',
                    bbox_inches = 'tight', dpi = 300)
        plt.close()
        
#%%

"""
    AFTER NOW PLOTS RELATED WITH THE Q (QUALITY FUNCTION)
"""


Q = pd.read_csv('results/Q_values.csv', index_col = 0)
Q.columns = Q.columns.astype(int)

F_states = min([k for (x, y, open, pos), k in CODE_STATES.items() if pos])
N_states = len(CODE_STATES)
N_actions = len(CODE_ACTIONS)

#%%

Values = pd.DataFrame(Q.max(axis = 1),
                      columns = ['value'],
                      index = range(F_states, N_states))
Values['state'] = Values.index

fig, _ = draw_values(
    Values,
    'state',
    'value',
    possession = True,
    open_play = True,
    interval = [0.1, 1],
    rotate_text = True,
    fontsize = 15
)
fig.savefig(os.path.join('figures/values', 'V_open_play.png'), dpi = 300)

fig, _ = draw_values(
    Values,
    'state',
    'value',
    possession = True,
    open_play = False,
    interval = [0.1, 1],
    rotate_text = True,
    fontsize = 15
)
fig.savefig(os.path.join('figures/values', 'V_set_piece.png'), dpi = 300)

# %%

# Best action on each state

best_act = pd.DataFrame(Q.idxmax(axis = 1),
                        columns = ['action'],
                        index = range(F_states, N_states))
best_act['state'] = Values.index

fig, _ = draw_action(
    best_act,
    'state',
    'action',
    possession = True,
    open_play = True
)
fig.savefig('figures/actions/' + \
            f'best_actions_open_play.png',
            bbox_inches = 'tight', dpi = 300)

fig, _ = draw_action(
    best_act,
    'state',
    'action',
    possession = True,
    open_play = False
)
fig.savefig('figures/actions/' + \
            f'best_actions_set_piece.png',
            bbox_inches = 'tight', dpi = 300)

# Now, short factor for facilitanting the visualization
short_factor = [-1, 5, 1.1]

fig, _ = draw_action(
    best_act,
    'state',
    'action',
    possession = True,
    open_play = True,
    short_factor = short_factor
)
fig.savefig('figures/actions/' + \
            f'best_actions_short_open_play.png',
            bbox_inches = 'tight', dpi = 300)

fig, _ = draw_action(
    best_act,
    'state',
    'action',
    possession = True,
    open_play = False,
    short_factor = short_factor
)
fig.savefig('figures/actions/' + \
            f'best_actions_short_set_piece.png',
            bbox_inches = 'tight', dpi = 300)


#%%

"""
    Quality Q(s, a) for particular states
"""

Q = pd.read_csv('results/Q_values.csv', index_col = 0)
Q.columns = Q.columns.astype(int)

F_states = min([k for (x, y, open, pos), k in CODE_STATES.items() if pos])
N_states = len(CODE_STATES)
N_actions = len(CODE_ACTIONS)

Q_df = Q.reset_index().melt(
    id_vars = 'index', var_name = 'action', value_name = 'quality')
Q_df.rename(columns = {'index': 'state'}, inplace = True)


#%%

def draw_some_actions_this_file(df, state_to_see, actions_to_see, suffix = ''):
    """
        Call other functions to draw a particular plot.
    """
    
    x, y, open, poss = decodify_states(state_to_see, copy = False)
    Q_act_to_see = Q_df.query('state == @state_to_see and action in @actions_to_see').copy()
    # next_state juts for where to show the number (action -> next_state)
    decoded_actions = np.array([DECODE_ACTIONS.get(action)
                                for action in Q_act_to_see['action']])
    for i, row in enumerate(decoded_actions):
        if row[0] == 43:
            decoded_actions[i, 1] = x + row[1]
            decoded_actions[i, 2] = y + row[2]
        if row[0] == 16:
            decoded_actions[i, 1] = GRID_WIDTH - 1
            decoded_actions[i, 2] = int(GRID_HEIGHT / 2)
    Q_act_to_see['next_succesfull_state'] = codify_states(
        decoded_actions[:, 1],
        decoded_actions[:, 2],
        True, True)
    Q_act_to_see['all_true'] = True

    fig, ax = draw_action(
        Q_act_to_see,
        'state',
        'action',
    )

    fig, ax = draw_values(
        Q_act_to_see,
        col_state = 'next_succesfull_state',
        col_value = 'quality',
        #interval = [None, 1],
        rotate_text = True,
        col_highlight = 'all_true',
        boxed_background = True,
        color_highlight = '#d3d3d3', # soft gray
        boxed_highlight = True,
        fig = fig,
        ax = ax,
        show_legend = False,
        color_box = False
    )

    fig.savefig('figures/actions/' + \
                f'state_{state_to_see}_quality_actions_{suffix}.png',
                bbox_inches = 'tight', dpi = 300)
    
#%%

# Now, see the best in the corner
print('Preparing images of actions after corner.')

x = 12; y = 0; open = False; poss = True
state_to_see = codify_states(x, y, open, poss)
print(f'State: ({x}, {y}, {open}, {poss}): {state_to_see}')

# List of actions to show
actions_to_see = [CODE_ACTIONS.get((30, xa, ya))
                    for xa in range(8, GRID_WIDTH)
                    for ya in range(GRID_HEIGHT)]
print(f'Actions: {actions_to_see}')

draw_some_actions_this_file(Q_df, state_to_see, actions_to_see, 'pass')

# List of actions to show - carry, shot
actions_to_see = [CODE_ACTIONS.get(act)
                    for act in [(16, 0, 0),
                                (43, 0, 1),
                                (43,-1, 1),
                                (43,-1, 0)]]
print(f'Actions: {actions_to_see}')

draw_some_actions_this_file(Q_df, state_to_see, actions_to_see, 'other')

#%%

# Now, see the best in a normal action
print('Preparing images of actions after corner.')

x = 9; y = 2; open = True; poss = True
state_to_see = codify_states(x, y, open, poss)
print(f'State: ({x}, {y}, {open}, {poss}): {state_to_see}')

# List of actions to show - passes
actions_to_see = [CODE_ACTIONS.get((30, xa, ya))
                    for xa in range(8, GRID_WIDTH)
                    for ya in range(GRID_HEIGHT)]
print(f'Actions: {actions_to_see}')

draw_some_actions_this_file(Q_df, state_to_see, actions_to_see, 'pass')

# List of actions to show - carry, shot
actions_to_see = [CODE_ACTIONS.get(act)
                    for act in [(16, 0, 0),
                                (43, 0, 1),
                                (43, 1, 1),
                                (43, 1, 0),
                                (43, 1,-1),
                                (43, 0,-1),
                                (43,-1,-1),
                                (43,-1, 0),
                                (43,-1, 1)]]
print(f'Actions: {actions_to_see}')

draw_some_actions_this_file(Q_df, state_to_see, actions_to_see, 'other')


#%%

# Now, see the best in a normal action
print('Preparing images of actions after corner.')

x = 9; y = 2; open = False; poss = True
state_to_see = codify_states(x, y, open, poss)
print(f'State: ({x}, {y}, {open}, {poss}): {state_to_see}')

# List of actions to show - passes
actions_to_see = [CODE_ACTIONS.get((30, xa, ya))
                    for xa in range(8, GRID_WIDTH)
                    for ya in range(GRID_HEIGHT)]
print(f'Actions: {actions_to_see}')

draw_some_actions_this_file(Q_df, state_to_see, actions_to_see, 'pass')

# List of actions to show - carry, shot
actions_to_see = [CODE_ACTIONS.get(act)
                    for act in [(16, 0, 0),
                                (43, 0, 1),
                                (43, 1, 1),
                                (43, 1, 0),
                                (43, 1,-1),
                                (43, 0,-1),
                                (43,-1,-1),
                                (43,-1, 0),
                                (43,-1, 1)]]
print(f'Actions: {actions_to_see}')

draw_some_actions_this_file(Q_df, state_to_see, actions_to_see, 'other')

# %%


