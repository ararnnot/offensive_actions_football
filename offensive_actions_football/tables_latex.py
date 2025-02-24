
import pandas as pd                                    

ARROWS_LATEX = {
    ( 1,-1) : r'$\nearrow$',
    ( 1, 0) : r'$\rightarrow$',
    ( 1, 1) : r'$\searrow$',
    ( 0,-1) : r'$\uparrow$',
    ( 0, 1) : r'$\downarrow$',
    (-1,-1) : r'$\nwarrow$',
    (-1, 0) : r'$\leftarrow$',
    (-1, 1) : r'$\swarrow$'
}

def table_san(data):

    df_print = pd.DataFrame(
        columns = ['time', 'player', 'state', 'action', 'next_state'],
        dtype = str
    )
    for i, row in data.iterrows():
        
        time = f'{(row["period"] - 1) * 45 +  int(row["time"]) // 60}:{int(row["time"]) % 60:02d}'
        player = row['player_name']
        
        state = f'({row["x"]}, {row["y"]}, ' + \
                ('open play, ' if row['open'] else 'set piece, ') + \
                ('maintain' if row['pos'] else 'change') + \
                ')'
        
        next_state = f'({row["x_next"]}, {row["y_next"]}, ' + \
                     ('open play, ' if row['open_next'] else 'set piece, ') + \
                     ('maintain' if row['pos_next'] else 'change') + \
                     ')'
        
        if row['action_type'] == 16:
            action = 'shot'
            next_state = 'goal' if row['reward'] else 'no goal'
        elif row['action_type'] == 30:
            action = f'pass({row["x_act"]}, {row["y_act"]})'
        elif row['action_type'] == 43:
            action = f'carry({ARROWS_LATEX[(row["x_act"], row["y_act"])]})'
            
        df_print.loc[len(df_print)] = [time, player, state, action, next_state]
        
    return df_print

def table_values(data):

    df_print = pd.DataFrame(
        columns = ['player_name', 'action', 'value', 'quality', 'value_next', 'value_increase', 'value_more_q'],
        dtype = str
    )
    for i, row in data.iterrows():
        
        if row['action_type'] == 16:
            action = 'shot'
            next_state = 'goal' if row['reward'] else 'no goal'
        elif row['action_type'] == 30:
            action = f'pass({row["x_act"]}, {row["y_act"]})'
        elif row['action_type'] == 43:
            action = f'carry({ARROWS_LATEX[(row["x_act"], row["y_act"])]})'
        
        value = f'{row["value"]:.3f}'
        quality = f'{row["quality"]:.3f}'
        value_next = f'{row["value_next"]:.3f}'
        value_increase = f'{row["value_increase"]:.3f}'
        value_more_q = f'{row["value_more_q"]:.3f}'
            
        df_print.loc[len(df_print)] = [
            row['player_name'], action, value, quality, value_next, value_increase, value_more_q
        ]
        
    return df_print
