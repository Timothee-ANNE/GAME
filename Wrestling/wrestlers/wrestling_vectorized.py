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
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
# // import envs from the envs folder and register them
import envs
import numpy as np
from clip import CLIP
from jax import jit, vmap
import jax.numpy as jnp
from sample_robot import sample_robot
import os
import matplotlib.pyplot as plt 

import jax
import jax.numpy as jnp
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


# %%
def make_env(env_id, blue_wrestler, red_wrestler, seed=None, render_mode=None, resolution=(224, 224)):
    """
    Creates a callable function that will create and return an environment
    with the specified parameters.
    """
    def _init():
        env = gym.make(env_id, blue_wrestler=blue_wrestler, red_wrestler=red_wrestler, seed=seed, render_mode=render_mode, resolution=resolution, world_path=os.path.join('world_data', 'flat.json'))
        env.metadata['render_fps'] = 24
        return env
    return _init


# %% [markdown]
# # config

# %%
materials = {
    0: "empty", # inisible
    1: "rigid",  # black 
    2: "soft",  # gray
    3: "horizontal",  # orange
    4: "vertical",  # blue
}

# %%
# Number of environments to run in parallel
num_envs = 10
n_steps = 200
frames_samples = list(np.linspace(0, n_steps-1, 6, dtype=int)[1:])
width = 5
seed = 42
rng = np.random.default_rng(seed)

# %%
blue_wrestlers = []
red_wrestlers = []
for i in range(num_envs):
    blue_wrestler, red_wrestler = sample_random_wrestler(rng, width)
    blue_wrestlers.append(blue_wrestler)
    red_wrestlers.append(red_wrestler)

# %%
env_fns = [make_env('WrestlingEnv-v0', blue_wrestlers[i], red_wrestlers[i], seed=0, render_mode=None) for i in range(num_envs)]
vec_env = SyncVectorEnv(env_fns) 

# %%
# run 
t0 = time()
vec_env.reset()
rgb_arrays = np.empty((num_envs, len(frames_samples), 224, 224, 3), dtype=np.uint8)  # CLIP resizes to 224x224 anyway
Observations = np.empty((num_envs, n_steps, 2))
for t in range(n_steps):  
    observations, _, dones, truncated, _ = vec_env.step([None] * num_envs)
    Observations[:, t] = observations
    if t in frames_samples:
        Quads = vec_env.call("get_quadrilaterals") 
        for i in range(num_envs):
            rgb_arrays[i, frames_samples.index(t)] = get_image(Quads[i])

#vec_env.close()
print((time()-t0)/num_envs)

# %%
colors


# %%
def closest_through_time(o):
    f_r = np.mean(np.argmin(np.abs(o-16), 1))
    f_b = 1-f_r
    return f_b, f_r


# %%
[closest_through_time(Observations[i]) for i in range(num_envs)]

# %%
plt.imshow(rgb_arrays[8,1])

# %%
