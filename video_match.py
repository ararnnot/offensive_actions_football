#%% Install dependencies

import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import io
from tqdm import tqdm

from offensive_actions_football.field_play import draw_state_actions

#%% Load data

data_path = os.path.join('data', 'data_steps.csv')
data = pd.read_csv(data_path)

#%% Video

frames = []
team_A = True

for i in tqdm(range(0, 500)):
    
    start = i - 8 if i - 8 >= 0 else 0
    fig, ax, team_A = draw_state_actions(
        data[start:i], alpha_increase = True, team_A = team_A, init = i)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    
    buf.seek(0)
    frame = Image.open(buf)
    frames.append(frame)

gif_path = "figures/matches/match_1.gif"
frames[0].save(gif_path, save_all = True, append_images = frames[1:],
               duration = 400, loop = 0)
print(f'Gif saved in {gif_path}')

#%% 

# Image case



# %%
