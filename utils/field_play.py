
import pandas as pd
import numpy as np

from matplotlib.colors import Normalize
from matplotlib.patches import RegularPolygon, Circle, Rectangle

from config.config_field import GRID_HEIGHT, GRID_WIDTH, \
                                FIELD_HEIGHT, FIELD_WIDTH
from utils.states_actions import decodify_states, DECODE_ACTIONS
from utils.field import get_centers, get_centers_int
from utils.field_plots import draw_pitch
from utils.field_draws import get_point_at_distance, draw_line, \
                                draw_own_arrow, draw_circle, \
                                draw_soccer_ball, draw_flag

def draw_state_actions(df, fig = None, ax = None,
                       col_states = 'state', col_actions = 'action',
                       col_next_states = 'next_state', col_rewards = 'reward',
                       team_A = True, alpha_increase = False, add_index = False,
                       init = 0, scoreboard = None, value_col = None):
    """
    Draw the field with a sequence of states and actions of the dataframe.
    init: initial integer index of the datafrem used for video coherence.
    """
    
    df = df.copy().reset_index(drop = True)
    
    if ax is None:
        fig, ax = draw_pitch(color = 'white', background = 'green')
        
    if alpha_increase and len(df) > 3:
        alpha = np.linspace(0.05, 0.8, len(df), endpoint = True)
    else:
        alpha = np.full(len(df), 0.8)
        
    team_A_next = team_A    
    
    #draw_separators(ax = ax, alpha = 0.1)
        
    for i, row in df.iterrows():
        
        x, y, open_play, possession = decodify_states(row[col_states], copy = False)
        xn, yn, open_play_n, possession_n = decodify_states(row[col_next_states], copy = False)
        action_type, xa, ya = DECODE_ACTIONS.get(row[col_actions])
        reward = row[col_rewards]
        
        color = 'blue' if team_A else 'red'
        
        if action_type == 16:
            
            x, y = get_centers_int(x, y)
            xa, ya = get_point_at_distance(x, y, FIELD_WIDTH, FIELD_HEIGHT // 2,
                                           distance = 2)
            if not team_A:
                x, y = FIELD_WIDTH - x, FIELD_HEIGHT - y
                xa, ya = FIELD_WIDTH - xa, FIELD_HEIGHT - ya
            draw_line(ax, x, y, xa, ya, color = color, alpha = alpha[i])
            
        elif action_type == 30:
            
            xa, ya = get_centers_int(xa, ya)
            x, y = get_centers_int(x, y)
            
            if not team_A:
                x, y = FIELD_WIDTH - x, FIELD_HEIGHT - y
                xa, ya = FIELD_WIDTH - xa, FIELD_HEIGHT - ya
            
            if xa == x and ya == y:
                draw_own_arrow(ax, x, y, color = color,
                               alpha = alpha[i], angle = init + i)
            else:
                draw_line(ax, x, y, xa, ya, color = color, dahsed = True, alpha = alpha[i])
            
        elif action_type == 43:
            
            xa, ya = get_centers_int(x+xa, y+ya)
            x, y = get_centers_int(x, y)
            
            if not team_A:
                x, y = FIELD_WIDTH - x, FIELD_HEIGHT - y
                xa, ya = FIELD_WIDTH - xa, FIELD_HEIGHT - ya
            draw_line(ax, x, y, xa, ya, color = color,
                      zigzag = True, small_zigzag = True, alpha = alpha[i])
        
        xn, yn = get_centers_int(xn, yn)
        if not possession_n:
            xn, yn = FIELD_WIDTH - xn, FIELD_HEIGHT - yn
        if not team_A:
            xn, yn = FIELD_WIDTH - xn, FIELD_HEIGHT - yn
        
        if not open_play:
            draw_flag(ax, x, y, color = color, alpha = alpha[i])
        if i == 0 and not possession_n:
            team_A_next = not team_A_next

        if (not np.isclose(xa, xn) or not np.isclose(ya, yn)) \
                and (i != len(df) - 1):
            draw_line(ax, xa, ya, xn, yn,
                      color = color, arrow = False,
                      dotted = True, alpha = alpha[i]*0.4)
        
        if add_index:
            bbox = dict(facecolor = 'white',
                        edgecolor = 'black',
                        boxstyle = 'circle, pad = 0.3')
            ax.text(x + FIELD_WIDTH // GRID_WIDTH // 2.5,
                    y - FIELD_HEIGHT // GRID_HEIGHT // 2.5,
                    str(i+1),
                    fontsize = 13, color = 'black', bbox = bbox,
                    ha = 'center', va = 'center', rotation = 0)
        
        if reward == 1:
            if team_A: xg, yg = FIELD_WIDTH, FIELD_HEIGHT // 2
            else: xg, yg = 0, FIELD_HEIGHT // 2
            draw_soccer_ball(ax, xg, yg, color = color, alpha = alpha[i])
        if not possession_n:
            team_A = not team_A
            if reward != 1:
                ax.scatter(xa, ya, s = 75, color = color, marker = 'x', lw = 2.2, alpha = alpha[i])
        
    if scoreboard is not None:
        x_coord = FIELD_WIDTH // 2 if value_col is None else FIELD_WIDTH // 4
        ax.text(x_coord - 5, - 5, scoreboard[0],
                fontsize = 20, color = 'blue', ha = 'center', va = 'center')
        ax.text(x_coord, - 5, '-',
                fontsize = 20, color = 'black', ha = 'center', va = 'center')
        ax.text(x_coord + 5, - 5, scoreboard[1],
                fontsize = 20, color = 'red', ha = 'center', va = 'center')
    
    if value_col is not None and not df.empty:
        value = df[value_col].iloc[-1]
        x_coord = FIELD_WIDTH // 2 if scoreboard is None else 3 * FIELD_WIDTH // 4
        max_length = FIELD_WIDTH // 5
        ax.plot([x_coord - max_length, x_coord + max_length], [-5, -5],
                color = 'white', lw = 2, alpha = 0.5, zorder = 1)
        rect = Rectangle((x_coord, -3), value * max_length, -4,
                            color = 'blue' if value > 0 else 'red', zorder = 2)
        ax.add_patch(rect)
        ax.plot([x_coord, x_coord], [-2, -8], color = 'black', lw = 3, zorder = 3)
            
    if df.empty: open_play_n = True
                  
    return fig, ax, team_A_next, open_play_n

def mid_point(x1, y1, x2, y2): return (x1 + x2) / 2, (y1 + y2) / 2
def orthonoal(x, y, l): return -y / np.linalg.norm([x, y]) * l, x / np.linalg.norm([x, y]) * l

def draw_state_action_values(df, fig = None, ax = None,
                             col_states = 'state', col_actions = 'action',
                             col_next_states = 'next_state', col_rewards = 'reward',
                             col_text = 'value_increase', flip_colors = False,
                             boxed_background = True, color_box = None,
                             txt_size = 10, rotate = False):
    
    fig, ax, _, _ = draw_state_actions(
        df[[col_states, col_actions, col_next_states, col_rewards]], 
        fig = fig,
        ax = ax,
        col_states = col_states,
        col_actions = col_actions,
        col_next_states = col_next_states,
        col_rewards = col_rewards,
        team_A = not flip_colors,
        alpha_increase = False,
        init = 0
    )
    
    team_A = not flip_colors
    
    for i, row in df.iterrows():
        
        x, y, open_play, possession = decodify_states(row[col_states], copy = False)
        xn, yn, open_play_n, possession_n = decodify_states(row[col_next_states], copy = False)
        action_type, xa, ya = DECODE_ACTIONS.get(row[col_actions])
        
        if action_type == 16:
            x, y = get_centers_int(x, y)
            xa, ya = get_point_at_distance(x, y, FIELD_WIDTH, FIELD_HEIGHT // 2,
                                           distance = 2)
        if action_type == 43:
            xa, ya = get_centers_int(x+xa, y+ya)
            x, y = get_centers_int(x, y)
        if action_type == 30:
            xa, ya = get_centers_int(xa, ya)
            x, y = get_centers_int(x, y)
            
        if not team_A:
            x, y = FIELD_WIDTH - x, FIELD_HEIGHT - y
            xa, ya = FIELD_WIDTH - xa, FIELD_HEIGHT - ya
        
        if i % 2 == 0: sign = -1
        else: sign = 1
        mid_x, mid_y = mid_point(x, y, xa, ya)
        orth_x, orth_y = orthonoal(xa - x, ya - y, 5 * sign)
        if mid_y >= 30 and mid_y <= 50 and mid_x >= 100:
            mid_x -= 5
        pos_x, pos_y = mid_x + orth_x, mid_y + orth_y

        bbox = dict(facecolor = 'white' if boxed_background else 'none',
                    edgecolor = color_box if color_box is not None else 'none',
                    boxstyle = 'round, pad = 0.2')
        
        angle = np.degrees(np.arctan2(orth_x, orth_y))
        if rotate:
            if angle < 0: angle += 180
        else:
            if angle > 90: angle -= 180
            if angle < -90: angle += 180
        txt = row[col_text]
        txt = f' + {txt:.3f}' if txt >= 0 else f' - {np.abs(txt):.3f}'
        
        ax.text(pos_x, pos_y,
                txt,
                ha = 'center', va = 'center',
                bbox = bbox,
                color = 'black' if row[col_text] >= 0 else 'red',
                fontsize = txt_size,
                rotation = angle)
        
        if not possession_n:
            team_A = not team_A
            
    return fig, ax
            
            