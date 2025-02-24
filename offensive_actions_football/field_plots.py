
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib import colormaps as cm

from matplotlib.colors import Normalize, PowerNorm

from config.config_field import GRID_HEIGHT, GRID_WIDTH, FIELD_HEIGHT, FIELD_WIDTH
from offensive_actions_football.states_actions import decodify_states, DECODE_ACTIONS, \
                                    codify_states, codify_actions
from offensive_actions_football.field import get_centers, get_centers_int
from offensive_actions_football.field_draws import draw_line, get_point_at_distance, \
                                draw_own_arrow

def draw_pitch(fig = None, ax = None, color = 'black', background = 'white'):
        
    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(1, 1, 1)
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    # Pitch Outline & Centre Line
    ax.plot([0, 120], [0, 0], color=color)
    ax.plot([0, 0], [0, 80], color=color)
    ax.plot([120, 120], [0, 80], color=color)
    ax.plot([0, 120], [80, 80], color=color)
    ax.plot([60, 60], [0, 80], color=color)

    # Left Penalty Area
    ax.plot([18, 18], [18, 62], color=color)
    ax.plot([0, 18], [62, 62], color=color)
    ax.plot([0, 18], [18, 18], color=color)

    # Right Penalty Area
    ax.plot([102, 120], [62, 62], color=color)
    ax.plot([102, 120], [18, 18], color=color)
    ax.plot([102, 102], [18, 62], color=color)

    # Left 6-yard Box
    ax.plot([6, 6], [30, 50], color=color)
    ax.plot([0, 6], [50, 50], color=color)
    ax.plot([0, 6], [30, 30], color=color)

    # Right 6-yard Box
    ax.plot([114, 120], [50, 50], color=color)
    ax.plot([114, 120], [30, 30], color=color)
    ax.plot([114, 114], [30, 50], color=color)

    # Left Goal
    ax.plot([-2, 0], [36, 36], color=color)
    ax.plot([-2, 0], [44, 44], color=color)
    ax.plot([-2, -2], [36, 44], color=color)

    # Right Goal
    ax.plot([120, 122], [36, 36], color=color)
    ax.plot([120, 122], [44, 44], color=color)
    ax.plot([122, 122], [36, 44], color=color)

    # Centre Circle
    centre_circle = plt.Circle((60, 40), 8, color=color, fill=False)
    centre_spot = plt.Circle((60, 40), 0.8, color=color)
    ax.add_patch(centre_circle)
    ax.add_patch(centre_spot)

    # Left Penalty Spot
    left_pen_spot = plt.Circle((12, 40), 0.8, color=color)
    ax.add_patch(left_pen_spot)

    # Right Penalty Spot
    right_pen_spot = plt.Circle((108, 40), 0.8, color=color)
    ax.add_patch(right_pen_spot)
    
    # Left Penalty Arc
    left_arc = patches.Arc((12, 40), height=20, width=20,
                           angle=0, theta1=308, theta2=52, color=color)
    ax.add_patch(left_arc)

    # Right Penalty Arc
    right_arc = patches.Arc((108, 40), height=20, width=20,
                            angle=0, theta1=128, theta2=232, color=color)
    ax.add_patch(right_arc)

    ax.set_xlim(-5, 125)
    ax.set_ylim(-10, 85)
    ax.set_aspect(1)
    ax.axis('off')

    ax.invert_yaxis()

    return fig, ax

def draw_two_pitches():
    
    # Not working well (!)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    draw_pitch(fig, axes[0])
    draw_pitch(fig, axes[1])

    return fig, axes

def draw_separators(fig = None, ax = None, alpha = 0.2):
    
    if ax is None:
        fig, ax = draw_pitch()
      
    for i in range(GRID_HEIGHT - 1):
        x, y = get_centers_int(0, i)
        xn, yn = get_centers_int(0, i+1)
        ax.plot([0, 120], [(y+yn)/2]*2,
                    color = "black", linestyle = "--", alpha = alpha)
    for j in range(GRID_WIDTH - 1):
        x, y = get_centers_int(j, 0)
        xn, yn = get_centers_int(j+1, 0)
        ax.plot([(x+xn)/2]*2, [0, 80],
                    color = "black", linestyle = "--", alpha = alpha)   


def draw_values(values, col_state, col_value,
                col_highlight = None, interval = [-1, 1],
                state_highlight_all = None, action_highlight_all = None,
                boxed_highlight = True, boxed_background = False,
                color_highlight = 'black', color_box = True,
                rotate_text = False, show_legend = True,
                rotate_legend = False, fontsize = 12,
                fig = None, ax = None,
                possession = True, open_play = True):
    
    values = values.copy()
    
    if ax is None:
        fig, ax = draw_pitch()
        
    if state_highlight_all is not None and action_highlight_all is not None:
        col_highlight = 'highlight'
        
        x, y, _, _ = decodify_states(state_highlight_all,
                                     copy = False)
        codifi_states = codify_states((x if possession else GRID_WIDTH - x - 1),
                                      (y if possession else GRID_HEIGHT - y - 1),
                                      open_play, possession)
        
        act, dx, dy = DECODE_ACTIONS[action_highlight_all]
        if act == 16: dx, dy = GRID_WIDTH-1, int((GRID_HEIGHT-1) / 2)
        if act == 43: dx, dy = x + dx, y + dy
        
        codifi_states_end = codify_states((dx if possession else GRID_WIDTH - dx - 1),
                                          (dy if possession else GRID_HEIGHT - dy - 1),
                                          open_play, possession)
        values['highlight'] = values[col_state].isin([codifi_states, codifi_states_end])
        
        x, y = get_centers_int(x, y)
        dx, dy = get_centers_int(dx, dy)
        if act == 16: draw_line(ax, x, y, dx, dy, color = 'black') 
        if act == 43: draw_line(ax, x, y, dx, dy, color = 'black', zigzag = True)
        if act == 30: draw_line(ax, x, y, dx, dy, color = 'black', dahsed = True)                       
            
        
    values['sector_x'], values['sector_y'], values['open_play'], values['possession'] \
        = decodify_states(values[col_state])
    
    values['center_x'], values['center_y'] \
        = get_centers(values['sector_x'], values['sector_y'])

    grid_data = np.full((GRID_HEIGHT, GRID_WIDTH), np.nan)

    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            row = values.query('sector_x == @j & sector_y == @i &' + \
                        'possession == @possession & open_play == @open_play')
            if not row.empty:
                grid_data[i, j] = row[col_value].iloc[0]
    if not possession:
        grid_data = np.flip(grid_data)

    if not color_box: grid_data = np.full((GRID_HEIGHT, GRID_WIDTH), np.nan)

    if interval == [-1,1]:
        cmap = cm.get_cmap('RdBu')
        norm = Normalize(vmin = interval[0], vmax = interval[1])
    if interval == [None, 1]:
        cmap = cm.get_cmap('YlGnBu')
        norm = PowerNorm(gamma = 0.35, vmin = 0, vmax = interval[1])
    if interval == [0.1, 1]:
        cmap = cm.get_cmap('YlGnBu')
        norm = PowerNorm(gamma = 0.48, vmin = 0, vmax = interval[1])
    if interval == [0,1]:
        cmap = cm.get_cmap('YlGnBu')
        norm = Normalize(vmin = interval[0], vmax = interval[1])
    
    img = ax.imshow(grid_data, cmap = cmap, norm = norm,
                    extent=(0, 120, 0, 80), origin='lower',
                    aspect='auto', alpha=0.6)
    
    cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    if interval == [None, 1]:
        cbar.set_ticks([0, 0.05, 0.2, 0.5, 1])
    if interval == [0.1, 1]:
        cbar.set_ticks([0, 0.1, 0.3, 0.6, 1])
    cbar.ax.tick_params(labelsize = fontsize)
    if rotate_legend: 
        cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), rotation = 90)
    if not show_legend: cbar.remove()

    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            row = values.query('sector_x == @j & sector_y == @i &' + \
                        'possession == @possession & open_play == @open_play')
            if not row.empty:
                
                row = row.iloc[0]
                
                bbox = None                
                if col_highlight is not None:
                    if row[col_highlight]:
                        bbox = dict(facecolor = 'white' if boxed_background else 'none',
                                    edgecolor = color_highlight if boxed_highlight else 'none',
                                    boxstyle = 'round, pad=0.2')
                        
                x_txt = row['center_x'] if possession else FIELD_WIDTH - row['center_x']
                y_txt = row['center_y'] if possession else FIELD_HEIGHT - row['center_y']
                if not rotate_text: y_txt -= FIELD_HEIGHT / (4 * GRID_HEIGHT)
                else: x_txt -= FIELD_WIDTH / (4 * GRID_WIDTH)
                ax.text(x_txt, y_txt,
                        f'{row[col_value]:.3f}',
                        ha='center', va='center',
                        bbox = bbox,
                        rotation = 0 if not rotate_text else 90,)
            
    draw_separators(ax = ax, alpha = 0.1)
    
    return fig, ax


def draw_action(data, col_state, col_action, col_color = None,
                alpha = 0.4, short_factor = None,
                fig = None, ax = None, background = 'white',
                possession = None, open_play = None):
    
    data = data.copy()
    
    if ax is None:
        fig, ax = draw_pitch(background = background)
    
    data['sector_x'], data['sector_y'], data['open_play'], data['possession'] \
        = decodify_states(data[col_state])
    
    if possession is not None and open_play is not None:
        data = data.query('possession == @possession & open_play == @open_play')
        
    for i, row in data.iterrows():
        
        act, xa, ya = DECODE_ACTIONS.get(row[col_action])
        x, y = row['sector_x'], row['sector_y']
        
        if act == 16:
            x, y = get_centers_int(x, y)
            xa, ya = get_point_at_distance(x, y,
                                           FIELD_WIDTH, FIELD_HEIGHT // 2,
                                           distance = 1)
            if short_factor is not None:
                if short_factor[0] == -1:
                    if x < FIELD_WIDTH - FIELD_WIDTH / GRID_WIDTH:
                        xa, ya = x + FIELD_WIDTH / GRID_WIDTH - 1, y + (ya - y) / 2
                else:
                    xa, ya = x + (xa - x) / short_factor[0], y + (ya - y) / short_factor[0]
            draw_line(ax, x, y, xa, ya, alpha = alpha,
                      color = 'red' if col_color is None else row[col_color])
        elif act == 30:
            x, y = get_centers_int(x, y)
            xa, ya = get_centers_int(xa, ya)
            if xa == x and ya == y:
                draw_own_arrow(ax, x, y,
                               alpha = alpha,
                               color = 'blue' if col_color is None else row[col_color])
            else:
                if short_factor is not None:
                    xa, ya = x + (xa - x) / short_factor[1], y + (ya - y) / short_factor[1]
                draw_line(ax, x, y, xa, ya,
                          dahsed = True, alpha = alpha,
                          color = 'blue' if col_color is None else row[col_color])
        elif act == 43:
            xa, ya = get_centers_int(x + xa, y + ya)
            x, y = get_centers_int(x, y)
            if short_factor is not None:
                xa, ya = x + (xa - x) / short_factor[2], y + (ya - y) / short_factor[2]
            draw_line(ax, x, y, xa, ya,
                      zigzag = True, alpha = alpha,
                      color = 'black' if col_color is None else row[col_color])
            
    draw_separators(ax = ax, alpha = 0.1)
    
    return fig, ax
         

