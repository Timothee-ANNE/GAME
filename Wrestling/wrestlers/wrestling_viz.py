# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # import 

# %%
import gymnasium as gym
from evogym import sample_robot
# // import envs from the envs folder and register them
import envs
import numpy as np

from tqdm import tqdm 
from PIL import Image, ImageDraw
import imageio
from sample_robot import sample_robot
from scipy.signal import convolve2d
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# %%
from time import time
from clip import get_image


# %% [markdown]
# # functions

# %%
def add_phase(rng, body):
    return body + np.where(body > 2, rng.integers(0, 2, size=body.shape)*2, 0.)

def sample_random_wrestler(rng, width):
    blue_wrestler = add_phase(rng, sample_robot(rng, (width, width)))
    red_wrestler = add_phase(rng, sample_robot(rng, (width, width)))
    return blue_wrestler, red_wrestler


# %% [markdown]
# # main

# %%
width = 5
seed = 42
mutation_probas = {"deletion": 1/3, "addition": 1/3, "mutation": 1/3}
n_mutations = 3
rng = np.random.default_rng(seed)

# %%
blue_wrestler, red_wrestler = sample_random_wrestler(rng, width)

# %%
red_wrestler = np.array([[0., 0., 0., 0., 0.],
          [5., 0., 0., 0., 0.],
          [4., 5., 0., 0., 0.],
          [1., 5., 5., 5., 5.],
          [5., 0., 0., 0., 0.]])

blue_wrestler = np.array([[0., 0., 5., 0., 4.],
          [0., 6., 1., 5., 4.],
          [0., 0., 0., 0., 4.],
          [5., 5., 5., 1., 4.],
          [0., 0., 0., 0., 5.]])

# %%
# // make the SimpleWalkingEnv using gym.make and with the robot information
env = gym.make('WrestlingEnv-v0', blue_wrestler=blue_wrestler, red_wrestler=red_wrestler, resolution=(720, 720),
               seed=None, render_mode="rgb_array", world_path=os.path.join('world_data', 'flat.json'))
env.metadata['render_fps'] = 24
obs, info = env.reset(seed=0)


# %%
T = 12
# // step the environment for 500 iterations
t0 = time()
frames = []
Obs = []
for i in tqdm(range(200)):
    obs, _, done, truncated, _ = env.step(None)
    print(done)
    Obs.append(obs)
    #Quads = env.get_wrapper_attr("get_quadrilaterals")()
    #frames.append(get_image(Quads))
    frames.append(env.render())
    #frames.append(env.get_wrapper_attr("get_rgb_array2")())
print(time()-t0)
#env.close()

# %%
s = 720/32
rectangles = [
    {
        'coords': (0, s*16-1, s*16, s*17),  # (x1, y1, x2, y2)
        'color': (32, 112, 180),  # Blue
        'outline': False  # Filled rectangle
    },
    {
        'coords': (s*16, s*16-1, s*32, s*17),  # (x1, y1, x2, y2)
        'color': (202, 23, 28),  # Red
        'outline': False  # Filled rectangle
    },
]

# %%
images = []
for i, rgb_array in enumerate(frames):
    img = Image.fromarray(np.array(rgb_array).astype('uint8'), 'RGB') 
    draw = ImageDraw.Draw(img)
    for rect in rectangles:
        coords = rect.get('coords')
        color = rect.get('color', (255, 0, 0))  # Default to red
        outline = rect.get('outline', False)
        width = rect.get('width', 1)
        if outline:
            draw.rectangle(coords, outline=color, width=width)
        else:
            draw.rectangle(coords, fill=color)
        radius = 5
        x = Obs[i][0]*s
        draw.ellipse((s+x-radius, 359+s/2-radius, s+x+radius, 359+s/2+radius), fill="black")
        x  = Obs[i][1]*s
        draw.ellipse((s+x-radius, 359+s/2-radius, s+x+radius, 359+s/2+radius), fill="black")
    images.append(img)
