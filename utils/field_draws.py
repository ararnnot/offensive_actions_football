
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.patches import RegularPolygon, Circle
    
def get_point_at_distance(x, y, end_x, end_y, distance = 5):
    
    v_x = x - end_x
    v_y = y - end_y
    magnitude = np.sqrt(v_x**2 + v_y**2)
    magnitude = 1e-5 if magnitude < 1e-5 else magnitude
    unit_v_x = v_x / magnitude
    unit_v_y = v_y / magnitude
    new_x = end_x + distance * unit_v_x
    new_y = end_y + distance * unit_v_y
    
    return new_x, new_y

def draw_line(ax, x, y, end_x, end_y,
               zigzag = False, small_zigzag = False,
               dahsed = False,
               arrow = True, dotted = False,
               color = 'black', alpha = 0.7):
    
    lw = 1.5 if dahsed else 2.5
    
    mid_x, mid_y = end_x, end_y
    if arrow:
        mid_x, mid_y = get_point_at_distance(x, y, end_x, end_y, distance = 3.9)
    if zigzag: 
        sketch = (3, 10, 1) if small_zigzag else (7, 25, 1)
        # sometimes the sketch work different. check the results
        #sketch = (10, 35, 1)
        with mpl.rc_context({'path.sketch': sketch}):
            ax.plot([x, mid_x], [y, mid_y], color = color, alpha = alpha)
    else:
        linestyle = (0, (5, 3)) if dahsed else '-' 
        if dotted:
            linestyle = (0, (0.8, 0.8))
        ax.plot([x, mid_x], [y, mid_y],
                color = color, alpha = alpha, linestyle = linestyle, lw = lw)
    
    if arrow:    
        mid_x, mid_y = get_point_at_distance(x, y, end_x, end_y, distance = 4)
        ax.annotate('', xy = (end_x, end_y), xytext = (mid_x, mid_y),
            arrowprops = dict(
                facecolor = color,
                edgecolor = color,
                lw = lw,
                arrowstyle='->, head_width=0.25, head_length=0.7',
                alpha = alpha
                )
        )

def bezier_curve(t, P0, P1, P2, P3):
    return (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3
       
def draw_own_arrow(ax, x, y, radius = 8, arrow = True,
                   color = 'blue', alpha = 0.7,
                   lw = 2, linestyle = (0, (6, 3)),
                   angle = None):
    
    if angle is None:
        angle = np.random.uniform(0, 2 * np.pi)
    
    t_values = np.linspace(0, 1, 100)
    bezier_points = np.array([bezier_curve(
        t,
        np.array([x + np.cos(angle) * radius / 4,
                  y + np.sin(angle) * radius / 4]),
        np.array([x + np.cos(angle) * radius,
                  y + np.sin(angle) * radius,]),
        np.array([x + np.cos(angle + np.pi/2) * radius,
                  y + np.sin(angle + np.pi/2) * radius]),
        np.array([x + np.cos(angle + np.pi/2) * radius / 4,
                  y + np.sin(angle + np.pi/2) * radius / 4]),
        ) for t in t_values])
    ax.plot(bezier_points[:, 0], bezier_points[:, 1],
            color = color, linestyle = linestyle, lw = lw, alpha = alpha)
    
    if arrow:
        mid_x, mid_y = get_point_at_distance(
            x + (0.2 * np.cos(angle) + np.cos(angle + np.pi/2)) * radius,
            y + (0.2 * np.sin(angle) + np.sin(angle + np.pi/2)) * radius,
            x + np.cos(angle + np.pi/2) * radius / 4,
            y + np.sin(angle + np.pi/2) * radius / 4,
            distance = 0.5)
        ax.annotate('', xy = (x, y), xytext = (mid_x, mid_y),
            arrowprops = dict(
                facecolor = color,
                edgecolor = color,
                lw = lw,
                arrowstyle='->, head_width=0.25, head_length=0.7',
                alpha = alpha
                )
        )
    
def draw_circle(ax, center_x, center_y, radius = 2, color = 'blue', alpha = 1):
    
    ball_outline = Circle((center_x, center_y), radius,
                          color=color, fill=False, lw=2, alpha = alpha)
    ax.add_patch(ball_outline)
    
def draw_soccer_ball(ax, center_x, center_y, radius = 2, color = 'blue', alpha = 1):
    
    ball_outline = Circle((center_x, center_y), radius, color=color,
                          fill=False, lw=2, alpha = alpha)
    ax.add_patch(ball_outline)
    
    pentagon = RegularPolygon((center_x, center_y), numVertices=5,
                              radius=radius * 0.5, 
                              orientation=np.pi/5, color=color, lw=1,
                              alpha = alpha)
    ax.add_patch(pentagon)
    
    for angle in np.linspace(-np.pi/2, 3*np.pi/2, 5, endpoint=False):
        ax.plot([center_x, center_x + np.cos(angle) * radius],
                [center_y, center_y + np.sin(angle) * radius],
                color = color, lw = 2, alpha = alpha
                )
    
    ax.text(center_x, center_y + radius * 2, 'GOAL!',
            color = color, ha = 'center', va = 'center')
    
def draw_flag(ax, x, y,
              flag_width = 4, flag_height = 3, pole_extra_height = 3,
              color="red", alpha = 1):
    """
    Draws a flag icon on a Matplotlib plot.
    """
    
    ax.plot([x, x], [y, y - pole_extra_height],
            color = color, linewidth = 2, alpha = alpha)
    
    flag = plt.Rectangle((x, y - pole_extra_height),
                         flag_width, - flag_height,
                         fill = False, edgecolor = color, alpha = alpha)
    ax.add_patch(flag)
    
    flag = plt.Rectangle((x, y - pole_extra_height - flag_height / 2),
                         flag_width / 2, - flag_height / 2, color = color, alpha = alpha)
    ax.add_patch(flag)
    
    flag = plt.Rectangle((x + flag_width / 2, y - pole_extra_height),
                         flag_width / 2, - flag_height / 2, color = color, alpha = alpha)
    ax.add_patch(flag)