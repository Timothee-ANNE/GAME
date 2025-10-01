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
# # Import 

# %%
import numpy as np
import os
from sklearn.cluster import KMeans
import pickle
from tqdm import tqdm
import matplotlib as mpl
mpl.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import multiprocess as mp
import queue
from mpl_toolkits.axes_grid1 import make_axes_locatable
from misc_plot import *
import utils
import subprocess
import pandas as pd
from time import time 
from copy import deepcopy
from functools import partial
import gc
import subprocess
import cv2
from PIL import Image, ImageDraw 
import umap.umap_ as umap  # pip install umap-learn
from scipy.spatial import cKDTree
from sklearn import preprocessing
import misc_plot
from scipy.signal import convolve2d

# %%
from GAME import *

# %%
import sys
sys.path.append("hearthbreaker")  
from hearthbreaker.constants import CHARACTER_CLASS

# %% [markdown]
# # Config 

# %%
hearthstone_config = {
    "blue_hero": CHARACTER_CLASS.WARRIOR,
    "red_hero": CHARACTER_CLASS.HUNTER,
    "n_replications": 50,
    "starting_side": "blue",
    "agent": "Trader",
    "timeout": 30,
    "exec_path": "PATH/TO/GAME/Hearthbreaker/run_games.py",
}

# %% [markdown]
# # Run

# %%
n_tasks = 1 # 50
budget = 40_000  # 20_000
seed_id = 0
n_cells = 1000 # 20

# %%
mini_config = {"n_tasks": n_tasks, "budget": budget, "seed_id": seed_id, "n_cells": n_cells, "hearthstone_config": hearthstone_config,}

# %% [markdown]
# ## GAME

# %%
# with random deck
if __name__ == "__main__":
    Classes = [
        (CHARACTER_CLASS.WARRIOR, CHARACTER_CLASS.WARLOCK),
        (CHARACTER_CLASS.ROGUE, CHARACTER_CLASS.PALADIN),
        (CHARACTER_CLASS.HUNTER, CHARACTER_CLASS.DRUID),
        (CHARACTER_CLASS.SHAMAN, CHARACTER_CLASS.MAGE),
        (CHARACTER_CLASS.MAGE, CHARACTER_CLASS.PRIEST),
    ]
    xps_index = []
    rng = np.random.default_rng(utils.seeds[0])

    for (red_hero, blue_hero) in Classes:
        for (opponent_class, hero_class) in [(red_hero, blue_hero), (blue_hero, red_hero)]:
            print(classes_names[hero_class], "vs", classes_names[opponent_class])
            hearthstone_config["blue_hero"] = opponent_class
            hearthstone_config["red_hero"] = hero_class
            config = get_config(hearthstone_config, budget, None, n_cells, "red", seed_id=seed_id)
            main_folder = utils.create_save_folder() 
            xps_index.append(f"{classes_names[hero_class]} vs {classes_names[opponent_class]} : '{main_folder}',")
            with open(main_folder + f"/xps_index.txt", "w") as f:
                f.writelines("\n".join(xps_index))     
            utils.save_pickle(main_folder + f"/mini_config.pk", mini_config)
            # init
            config["save_folder"] = main_folder
            utils.create_folder(config["save_folder"])
            
            config["tasks"] = create_tasks([{"deck": rng.choice(valid_cards_names[opponent_class], 30), "hero_class": opponent_class}], generation="red")  # means optimizing the reds and fixing the blues
            me = MT_GAME(config)
            # run gen
            me.run()
