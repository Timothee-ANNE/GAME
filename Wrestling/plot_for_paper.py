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
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocess as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable
from misc_plot import *
import utils
import subprocess
import pandas as pd
from time import time 
from copy import deepcopy
from functools import partial
import jax.numpy as jnp
import gc
import jax
import subprocess
import cv2
from PIL import Image, ImageDraw, ImageFont
import umap.umap_ as umap  # pip install umap-learn
from scipy.spatial import cKDTree
from sklearn import preprocessing
import plot
from matplotlib.figure import Figure
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from kmodes.kmodes import KModes

# %%
from GAME import *

# %%
if os.uname()[1] in ['tripper4', 'miami', 'dada']:
    N_PROC = 64
elif os.uname()[1] in ['mac624183 ', 'LXGPQQ7GDY']:
    N_PROC = 6
elif os.uname()[1] in ['Ooki']:
    N_PROC = 32
else:
    print("Unknown machine. N_PROC set to 1")
    N_PROC = 1


# %% [markdown]
# # Functions

# %%
def asymmetric_elo(tournament_matrix, iterations=5, k=32, initial_rating=1500):
    num_blue = tournament_matrix.shape[1]
    num_red = tournament_matrix.shape[0]
    
    # Initialize ratings
    blue_ratings = np.ones(num_blue) * initial_rating
    red_ratings = np.ones(num_red) * initial_rating
    
    for _ in range(iterations):
        # Calculate expected scores
        expected_scores = np.zeros((num_red,num_blue))
        for i in range(num_red):
            for j in range(num_blue):
                expected_scores[i,j] = 1 / (1 + 10**((red_ratings[i] - blue_ratings[j])/400))
        
        # Calculate adjustments
        blue_adjustments = np.zeros(num_blue)
        red_adjustments = np.zeros(num_red)
        
        for i in range(num_blue):
            blue_adjustments[i] = k * np.sum((1-tournament_matrix[:, i]) - (1-expected_scores[:, i]))
            
        for j in range(num_red):
            red_adjustments[j] = k * np.sum(tournament_matrix[j, :] - expected_scores[j, :])
        
        # Update ratings
        blue_ratings += blue_adjustments
        red_ratings += red_adjustments
        
    return red_ratings, blue_ratings


# %%
rgb_colors = {
    0.: np.array([255,255,255]),
    1.: np.array([1, 1, 1]),
    2.: np.array([211, 211, 211]),
    3.: np.array([255, 165, 0]),
    4.: np.array([65, 105, 225]),
    5.: np.array([255, 215, 0]),
    6.: np.array([135, 206, 235]),
}

def viz_wrestler(wrestler):
    image = np.ones((5,5,3))*255
    for i in range(5):
        for j in range(5):
            image[i,j] = rgb_colors[wrestler[i,j]]/255
    return image


# %% [markdown]
# # Main Figures 

# %% [markdown]
# ## Paths 

# %%
blue_color = "#2070b4"
red_color = "#ca171c" 

# %%
line_styles = ["-", (0, (1, 1)),   (0, (5, 1)),  (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5)),  (0, (5, 5)), ]
colors = cm.rainbow
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]

# %%
font = {'size'   : 18}
mpl.rc('font', **font)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# %%
Folders = {
    "GAME": [
        "/home/tim/Experiments/data/2025/07/19/01h25m39s/",
        "/home/tim/Experiments/data/2025/07/19/16h12m29s/",
        "/home/tim/Experiments/data/2025/07/21/16h53m40s/",
    ],
    "Random": [
        "/home/tim/Experiments/data/2025/07/22/14h53m32s/",
        "/home/tim/Experiments/data/2025/07/23/06h01m56s/",
        "/home/tim/Experiments/data/2025/07/23/21h04m09s/",
    ],
}

# %%
labels = {
    "GAME": "GAME",
    "Random": "Random",
}

# %% [raw]
# for variant, replications in Folders.items():
#     for folder in replications:
#         for i in range(10):
#             os.system(f"rm {folder}/gen_{i}/tournament_{i}.pk")

# %%
flatten_Folders = []
for variant, replications in Folders.items():
    for folder in replications:
        flatten_Folders.append((variant, folder))

# %% [markdown]
# ## Soft robots (main figures)

# %%
precomputes_folder = "/home/tim/Experiments/data/GAME/soft_robot/precomputes_with_random/"
figure_folder = "/home/tim/Experiments/data/GAME/soft_robot/figures/"

# %% [markdown]
# ### Load inter-gen tournament

# %%
F_pca, B_pca, F, indices, sol_PCA = [], [], {}, [], []
F_Obs = []

for variant, folders in Folders.items():
    for folder in folders:
        key = (variant, folder)
        print(key)
        n_gen = 10
        Elites = [utils.load_pickle(folder + f"/gen_{gen_id}/elites_{gen_id}.pk") for gen_id in range(n_gen)]
        
        blues, reds = [], []
        for gen_id in [i for i in range(1, n_gen, 2)]:
            for blue in Elites[gen_id]["blues"]:
                blues.append(blue)
        for gen_id in [i for i in range(0, n_gen, 2)]:
            for red in Elites[gen_id]["reds"]:
                reds.append(red)
        
        # compute red generation start (i) and end (j)
        i, red_indices = 0, {}
        for gen_id in range(0, n_gen, 2):
            n_reds = len(Elites[gen_id]["reds"])
            j = i + n_reds
            red_indices[gen_id] = (i,j)
            i = j 
        
        # compute blue generation start (i) and end (j)
        i, blue_indices = 0, {}
        for gen_id in range(1, n_gen, 2):
            n_blues = len(Elites[gen_id]["blues"])
            j = i + n_blues
            blue_indices[gen_id] = (i,j)
            i = j 
        
        n_red, n_blue = len(reds), len(blues)
        F_red = np.ones((n_red, n_blue))*0.5
        for blue_gen_id, (blue_i, blue_j) in blue_indices.items():
            for red_gen_id, (red_i, red_j) in red_indices.items():
                mini_tournament = utils.load_pickle(folder + f"generational_tournament_{blue_gen_id}_{red_gen_id}.pk")
                for (i, j), duel in mini_tournament.items():
                    F_red[i,j] = duel["eval"]["fitness"]
                    B_pca.append(np.array(duel["eval"]["behavior"]))
                    F_pca.append(duel["eval"]["fitness"])
                    indices.append((key, blue_gen_id, red_gen_id, i, j))
                    sol_PCA.append({"candidate": duel["candidate"], "task": duel["task"]})
                    F_Obs.append(duel["eval"]["obs"])
        F[key] = F_red

F_pca = np.array(F_pca)
F_Obs = np.array(F_Obs)
utils.save_pickle(precomputes_folder + "indices.pk", indices)
utils.save_pickle(precomputes_folder + "F_pca.pk", F_pca)
utils.save_pickle(precomputes_folder + "F.pk", F)

# %% [markdown]
# ### PCA

# %%
all_B = np.array(B_pca)
all_B /= np.linalg.norm(all_B, axis=1)[:, np.newaxis]
pca = PCA(n_components=2)
pca.fit(all_B)
projection = pca.transform(all_B)
utils.save_pickle(precomputes_folder + "projection.pk", projection)

# %%
pca.explained_variance_ratio_ # array([0.13438183, 0.08421282])

# %% [markdown]
# pca.explained_variance_ratio_ array([0.10553934, 0.08642663])

# %%
indices = utils.load_pickle(precomputes_folder + "indices.pk")
rev_indices = []
for name in list(flatten_Folders):
    rev_indices.append([i for i in range(len(indices)) if indices[i][0]==name])

# %%
plt.subplots(figsize=(4*3, 4*2))
plt.axis("off")

xmin, ymin = np.min(projection, axis=0)
xmax, ymax = np.max(projection, axis=0)
xpad = (xmax-xmin)*0.05
ypad = (ymax-ymin)*0.05
for i, (variant, folder) in enumerate(flatten_Folders):
    y = list(Folders).index(variant)
    x = Folders[variant].index(folder)
    plt.subplot2grid((2, 3*5+1), (y, 5*x), colspan=5)
    
    scatter = plt.scatter(projection[rev_indices[i], 0], projection[rev_indices[i], 1], s=2, c=F_pca[rev_indices[i]], vmin=0, vmax=1, alpha=0.2, cmap="coolwarm",)
    #plt.xlabel(f"PCA 0 ({100*pca.explained_variance_ratio_[0]:2.1f}%)")
    #plt.ylabel(f"PCA 1 ({100*pca.explained_variance_ratio_[1]:2.1f}%)")
    plt.xticks([]);
    plt.yticks([]);
    plt.xlim((xmin-xpad, xmax+xpad))
    plt.ylim((ymin-ypad, ymax+ypad))
    if x == 1:
        plt.title(list(Folders)[y], fontsize=34, va="center")

ax_colorbar = plt.subplot2grid((1, 3*7+1), (0, 3*7), rowspan=1, colspan=2)
norm = scatter.norm  # Get the normalization from the scatter
cbar = ColorbarBase(ax_colorbar, cmap=scatter.cmap, norm=norm, orientation='vertical')
cbar.set_label('Fitness', fontsize=44)

plt.tight_layout(pad=0)
#plt.savefig(f"{figure_folder}PCAs.png") 

# %% [markdown]
# ### Speed

# %%
i = 0
o = F_Obs[rev_indices[i]]

np.mean(np.abs(o[:, 1:] - o[:, :-1]), axis=(1,2))*2


# %%
plt.subplots(figsize=(4*3, 4*2))
plt.axis("off")

xmin, ymin = np.min(projection, axis=0)
xmax, ymax = np.max(projection, axis=0)
xpad = (xmax-xmin)*0.05
ypad = (ymax-ymin)*0.05
for i, (variant, folder) in enumerate(flatten_Folders):
    y = list(Folders).index(variant)
    x = Folders[variant].index(folder)
    plt.subplot2grid((2, 3*5+1), (y, 5*x), colspan=5)

    
    o = F_Obs[rev_indices[i]]
    speed = np.mean(np.abs(o[:, 1:] - o[:, :-1]), axis=(1,2)) * 25

    indices = np.argsort(speed[:])
    scatter = plt.scatter(projection[rev_indices[i], 0][indices], projection[rev_indices[i], 1][indices], s=2, c=speed[indices], vmin=0, vmax=2, alpha=0.1, cmap="viridis")
    #plt.xlabel(f"PCA 0 ({100*pca.explained_variance_ratio_[0]:2.1f}%)")
    #plt.ylabel(f"PCA 1 ({100*pca.explained_variance_ratio_[1]:2.1f}%)")
    plt.xticks([]);
    plt.yticks([]);
    plt.xlim((xmin-xpad, xmax+xpad))
    plt.ylim((ymin-ypad, ymax+ypad))
    if x == 1:
        plt.title(list(Folders)[y], fontsize=34, va="center")

ax_colorbar = plt.subplot2grid((1, 3*7+1), (0, 3*7), rowspan=1, colspan=2)
norm = scatter.norm  # Get the normalization from the scatter
cbar = ColorbarBase(ax_colorbar, cmap=scatter.cmap, norm=norm, orientation='vertical')
cbar.set_label('Speed (pixel/s)', fontsize=34)

plt.tight_layout(pad=0)
plt.savefig(f"{figure_folder}PCAs_speed.png") 


# %% [markdown]
# ### QD-Score

# %%
def compute_qd_score(behaviors, fitness, step_size=0.01, alpha=1., verbose=False):
    qd_score, coverage = np.empty(len(behaviors)), np.empty(len(behaviors))
    cells = {}
    for i in tqdm(range(len(behaviors))) if verbose else range(len(behaviors)):
        key = tuple([int(x) for x in behaviors[i]/step_size])
        if key not in cells:
            cells[key] = fitness[i]
        else:
            cells[key] = max(cells[key], fitness[i])
        qd_score[i] = sum((x for x in cells.values())) * step_size**len(key)
        coverage[i] = len(cells)* step_size**len(key)
    return qd_score, coverage, cells


# %%
measures = {}
standardized_projection = (projection - np.min(projection, axis=0))/(np.max(projection, axis=0) - np.min(projection, axis=0))
for i, name in tqdm(enumerate(flatten_Folders)):
    red_qd_score, coverage, red_cells = compute_qd_score(standardized_projection[rev_indices[i]], F_pca[rev_indices[i]], step_size=0.01, alpha=1., verbose=False)
    blue_qd_score, coverage, blue_cells = compute_qd_score(standardized_projection[rev_indices[i]], 1-np.array(F_pca[rev_indices[i]]), step_size=0.01, alpha=1., verbose=False)
    measures[name] = {"coverage": coverage, "red_qd_score": red_qd_score, "red_cells": red_cells, "blue_qd_score": blue_qd_score, "blue_cells": blue_cells}
utils.save_pickle(precomputes_folder + "measures.pk", measures)

# %%
measures = utils.load_pickle(precomputes_folder + "measures.pk")
indices = utils.load_pickle(precomputes_folder + "indices.pk")
Red_qd_score, Coverage, Blue_qd_score = {}, {}, {}
for name in flatten_Folders:
    idx_min = np.min( [i for i in range(len(indices)) if indices[i][0]==name])
    coverage, red_qd_score, blue_qd_score = [], [], []
    for blue_gen_id, red_gen_id in [(1,0), (1,2), (3,2), (3,4), (5,4), (5,6), (7,6), (7,8), (9,8)]:
        idx = np.max([i-idx_min for i in range(len(indices)) if indices[i][0]==name and indices[i][1]==blue_gen_id and indices[i][2]==red_gen_id])
        coverage.append(100 * measures[name]["coverage"][idx])
        red_qd_score.append(100 * measures[name]["red_qd_score"][idx])
        blue_qd_score.append(100 * measures[name]["blue_qd_score"][idx])
    Coverage[name] = coverage
    Red_qd_score[name] = red_qd_score
    Blue_qd_score[name] = blue_qd_score

# %%

# %%
data, names = [], []
for variant, replications in Folders.items():
    rep_data = []
    for folder in replications:
        key = (variant, folder)
        path = os.path.join(root_folder, variant, folder)
        rep_data.append(Coverage[key])
    data.append(rep_data)
    names.append(labels[variant])

plt.subplots(figsize=(8,4.5))

line_styles = ["-"] * len(data) if line_styles is None else line_styles
for i, Y in enumerate(data):
    median = np.median(Y, axis=0) 
    maxi = np.max(Y, axis=0) 
    mini = np.min(Y, axis=0) 
    print(f"{names[i]}: {median[-1]:2.1f} [{maxi[-1]:2.1f}, {mini[-1]:2.1f}]")
    color = colors[i] if type(colors) == list else int_to_color(i, len(data), cmap=colors)
    x = [i for i in range(0, 9)]
    plt.plot(x, median, lw=3, color=color, ls=line_styles[i], label=names[i])
    plt.fill_between(x, mini, maxi , color=color, alpha=0.3)
    plt.xlabel("Generations")
    plt.ylabel("Coverage (%)")
    plt.grid(axis="y", alpha=0.5)
    x_ticks = [f"{i}-{i+1}" for i in range(0,9)]
    plt.xticks(x, x_ticks)
#plt.legend(fontsize=10, handlelength=5)
plt.title("Visual Diversity")
plt.tight_layout()
plt.savefig(figure_folder + "wrestling_coverage.pdf")

# %%
kind = "Both"

data, names = [], []
for variant, replications in Folders.items():
    rep_data = []
    for folder in replications:
        key = (variant, folder)
        if kind == "Red":
            rep_data.append(np.array(Red_qd_score[key]))
        elif kind == "Blue":
            rep_data.append(np.array(Blue_qd_score[key]))
        else:
            rep_data.append((np.array(Blue_qd_score[key]) + np.array(Red_qd_score[key]))/2)
    data.append(rep_data)
    names.append(labels[variant])

plt.subplots(figsize=(8,4.5))
line_styles = ["-"] * len(data) if line_styles is None else line_styles
for i, Y in enumerate(data):
    median = np.median(Y, axis=0) 
    maxi = np.max(Y, axis=0) 
    mini = np.min(Y, axis=0) 
    print(f"{names[i]}: {median[-1]:2.1f} [{maxi[-1]:2.1f}, {mini[-1]:2.1f}]")
    color = colors[i] if type(colors) == list else int_to_color(i, len(data), cmap=colors)
    x = [i for i in range(0, 9)]
    plt.plot(x, median, lw=3, color=color, ls=line_styles[i], label=names[i])
    plt.fill_between(x, mini, maxi , color=color, alpha=0.3)
    plt.xlabel("Generations")
    plt.ylabel("QD-Score (%)")
    plt.grid(axis="y", alpha=0.5)
    plt.title("")
    x_ticks = [f"{i}-{i+1}" for i in range(0,9)]
    plt.xticks(x, x_ticks)
plt.legend(fontsize=10, handlelength=5)
plt.title("Quality Diversity "+kind)
plt.tight_layout()

# %% [markdown]
# ### ELO (inter variants)

# %%
all_F = utils.load_pickle(precomputes_folder + "F.pk")

# %%
Red_ratings, Blue_ratings = {}, {}
for variant, folder in flatten_Folders:
    name = (variant, folder)
    F = all_F[name]
    red_ratings, blue_ratings = asymmetric_elo(F)
    Red_ratings[name] = red_ratings
    Blue_ratings[name] = blue_ratings

# %%
n_elites = 10

# %%
Red_Elites, Blue_Elites = [], []

for variant, folders in Folders.items():
    for folder in folders:
        key = (variant, folder)
        n_gen = 10
        Elites = [utils.load_pickle(folder + f"/gen_{gen_id}/elites_{gen_id}.pk") for gen_id in range(n_gen)]
        
        blues, reds = [], []
        for gen_id in [i for i in range(1, n_gen, 2)]:
            for blue in Elites[gen_id]["blues"]:
                blues.append(blue)
        for gen_id in [i for i in range(0, n_gen, 2)]:
            for red in Elites[gen_id]["reds"]:
                reds.append(red)


# %%
Red_Elites, Blue_Elites = [], []
Red_Origines, Blue_Origines = [], []

for variant, folder in tqdm(flatten_Folders):
    name = (variant, folder)
    n_gen = 10
    Elites = [utils.load_pickle(folder + f"/gen_{gen_id}/elites_{gen_id}.pk") for gen_id in range(n_gen)]
    all_red_Elites, all_blue_Elites = [], []
    for gen_id in [i for i in range(1, n_gen, 2)]:
        for blue in Elites[gen_id]["blues"]:
            all_blue_Elites.append(blue)
    for gen_id in [i for i in range(0, n_gen, 2)]:
        for red in Elites[gen_id]["reds"]:
            all_red_Elites.append(red)

    assert len(all_red_Elites) == len(Red_ratings[name])
    assert len(all_blue_Elites) == len(Blue_ratings[name])

    red_Elites, blue_Elites = [], []
    red_Elites_txt, blue_Elites_txt = [], []
    red_Elites_origin, blue_Elites_origin = [], []
    for i in np.argsort(Red_ratings[name])[::-1]:
        bt = all_red_Elites[i]
        bt_txt = tuple(bt["wrestler"].reshape(-1))
        if bt_txt not in red_Elites_txt:
            red_Elites_txt.append(bt_txt)
            red_Elites.append(bt)
            red_Elites_origin.append(name +(i,))
        if len(red_Elites) == n_elites:
            break
    for i in np.argsort(Blue_ratings[name])[::-1]:
        bt = all_blue_Elites[i]
        bt_txt =  tuple(bt["wrestler"].reshape(-1))
        if bt_txt not in blue_Elites_txt:
            blue_Elites_txt.append(bt_txt)
            blue_Elites.append(bt)
            blue_Elites_origin.append(name +(i,))
        if len(blue_Elites) == n_elites:
            break
    Red_Elites += red_Elites
    Blue_Elites += blue_Elites
    Red_Origines += red_Elites_origin
    Red_Origines += blue_Elites_origin

# %%
tournaments_path = precomputes_folder 

# %%
utils.save_pickle(precomputes_folder + f'{6}_{10}_tournament_Folders.pk', (Folders, flatten_Folders))
config = get_config(wrestlers_config, budget, None, n_cells, use_diversity_only, use_quality_only, seed_id)
comparison_tournament = compute_tournament(config, Red_Elites, Blue_Elites)
utils.save_pickle(precomputes_folder + f'{6}_{10}_comparison_tournament.pk', comparison_tournament)

# %%
comparison_tournament = utils.load_pickle(precomputes_folder + f'{6}_{10}_comparison_tournament.pk')
(tournament_Folders, tournament_folders) = utils.load_pickle(precomputes_folder + f'{6}_{10}_tournament_Folders.pk')

# %%
n_red, n_blue = len(Red_Elites), len(Blue_Elites)
F = np.ones((n_red, n_blue))*0.5
for (i,j), val in comparison_tournament.items():
    F[i,j] = val["eval"]["fitness"]
    
red_ratings, blue_ratings = asymmetric_elo(F)

red_data = [[] for _ in range(2)] 
blue_data = [[] for _ in range(2)] 
both_data = []

for i, (variant, folder) in enumerate(tournament_folders):
    name = (variant, folder)
    j = list(tournament_Folders.keys()).index(variant)
    red_data[j].append(red_ratings[i*n_elites:(i+1)*n_elites])
    blue_data[j].append(blue_ratings[i*n_elites:(i+1)*n_elites])

for i in range(2):
    red_data[i] = np.concatenate(red_data[i])
    blue_data[i] = np.concatenate(blue_data[i])
    both_data.append(np.concatenate([red_data[i], blue_data[i]]))

# %%
txt = plot_boxplot(both_data, list(labels.values()),  title="", colors=colors,
             swarmsize=3, ylabel="ELO score", figsize=(6,3), rotation=0, use_stick=False, fontsize=18);
plt.savefig(figure_folder + "wrestling_ELO.pdf")

# %%
print(txt)

# %% [markdown]
# ## PCA video

# %%
plt.subplots(figsize=(4, 4))
plt.axis("off")

xmin, ymin = np.min(projection, axis=0)
xmax, ymax = np.max(projection, axis=0)
xpad = (xmax-xmin)*0.05
ypad = (ymax-ymin)*0.05
i = 0
(variant, folder) = flatten_Folders[i]

y = list(Folders).index(variant)
x = Folders[variant].index(folder)

scatter = plt.scatter(projection[rev_indices[i], 0], projection[rev_indices[i], 1], s=2, c=F_pca[rev_indices[i]], vmin=0, vmax=1, alpha=0.2, cmap="coolwarm",)
plt.xticks([]);
plt.yticks([]);
plt.xlim((xmin-xpad, xmax+xpad))
plt.ylim((ymin-ypad, ymax+ypad))

# %%
step_size = 0.1 

cells = {}
xmin, ymin = np.min(projection, axis=0)
xmax, ymax = np.max(projection, axis=0)
for duel_id in np.array(rev_indices[0])[np.random.permutation(len(rev_indices[0]))]:
    x, y = projection[rev_indices[0][duel_id]]
    x = (x-xmin)/(xmax-xmin)
    y = (y-ymin)/(ymax-ymin)
    i = int(x/step_size) if x != xmax else int(1/step_size-1)
    j = int(y/step_size) if y != ymax else int(1/step_size-1)
    if (i,j) not in cells:
        cells[i,j] = duel_id

# %%
plt.subplots(figsize=(10, 10))
plt.axis("off")

xmin, ymin = np.min(projection, axis=0)
xmax, ymax = np.max(projection, axis=0)
xpad = (xmax-xmin)*0.05
ypad = (ymax-ymin)*0.05
i = 0
(variant, folder) = flatten_Folders[i]

y = list(Folders).index(variant)
x = Folders[variant].index(folder)

scatter = plt.scatter(projection[rev_indices[i], 0], projection[rev_indices[i], 1], s=2, c=F_pca[rev_indices[i]], vmin=0, vmax=1, alpha=0.2, cmap="coolwarm",)
for duel_id in cells.values():
    plt.scatter(projection[duel_id, 0], projection[duel_id, 1], color="black", marker="+")
    pad = 0
    plt.text(projection[duel_id, 0]+pad, projection[duel_id, 1]+pad, str(duel_id), fontsize=10)
    
plt.xticks([]);
plt.yticks([]);
plt.xlim((xmin-xpad, xmax+xpad))
plt.ylim((ymin-ypad, ymax+ypad));

# %%
sol_id = list(cells.values())[0]

# %%
for sol_id in tqdm(cells.values()):
    duel = sol_PCA[sol_id]
    candidates = [duel["candidate"]]
    tasks = [duel["task"]]
    config = get_config(wrestlers_config, None, None, None)
    evaluation = evaluate_wrestlers_for_video(**config["evaluation_config"], resolution=(720, 720), tasks=tasks, candidates=candidates)

    o = evaluation[0]["obs"]
    ftt = fitness_throught_time(o)
    
    
    image_size = 720
    timestep = 1
    n_steps = 200
    W = image_size
    H = image_size//2
    
    images = []
    for t in range(n_steps//timestep):
        new = Image.new("RGBA", (W, H))
        draw = ImageDraw.Draw(new)
        new.paste(Image.new("RGBA", new.size, "WHITE"), (0,0))
        images.append(new)
    
    frames = evaluation[0]["rgb_array"]
    
    # Configuration for rectangle and text
    font_size = 50
    
    # Try to load a font (fallback to default if not available)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    for t in range(n_steps//timestep):
        (x_pos, y_pos) = i, j
        y = 0
        x = 0
        
        # Convert frame to PIL Image
        img = Image.fromarray(frames[t].astype(np.uint8)[:720//2])
        img = img.resize((image_size, image_size//2))
        
        # Create a drawing context
        draw = ImageDraw.Draw(img)
        
    
        for side in ["Red", "Blue"]:
            # Add text at center bottom
            text = f"{side} {np.sum(ftt[:t]) if side == 'Red' else np.sum(1-ftt[:t])}"  # Change this text as needed
            
            # Get text dimensions for centering
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Calculate position for center bottom
            text_x = (image_size - text_width) // 2 + (200 if side == 'Red' else -200)
            text_y = text_height + 10  
            
            # Draw text with background for better visibility
            draw.rectangle([text_x - 5, text_y - 5, text_x + text_width + 5, text_y + text_height + 20], 
                           fill=(255, 255, 255, 128))  # Semi-transparent black background
            draw.text((text_x, text_y), text, fill=blue_color if side == "Blue" else red_color, font=font)
            
            # Paste the modified image
            images[t].paste(img, (x, y))
            
    path = figure_folder + f'/{sol_id}'
    images[0].save( path + '.gif', save_all=True, append_images=images[1:], optimize=True, duration=25, loop=0)
    
    videodims = images[0].size
    fourcc = cv2.VideoWriter_fourcc(*'avc1')    
    video = cv2.VideoWriter(path + ".mp4", fourcc, 1000//25, videodims)
    img = Image.new('RGB', videodims, color = 'darkred')
    for image in images:
        video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    video.release()


# %%
def fitness_throught_time(o):
    f_r = np.argmin(np.abs(o-16), 1)
    return f_r


# %% [markdown]
# ## Soft Robots (individual)

# %% [markdown]
# ### get the best of one generation

# %%
gen_id = 1
gen_folder_path = root_folder +  f"gen_{gen_id}/"


# %%
tournament =  utils.load_pickle(gen_folder_path + f"tournament_{gen_id}.pk")
Elites = utils.load_pickle(gen_folder_path + f"elites_{gen_id}.pk")

blues = Elites["blues"]
reds = Elites["reds"]

n_red, n_blue = len(reds), len(blues)
F = np.zeros((n_red, n_blue))
for (i,j), val in tournament.items():
    f_r, f_b = val["eval"]["fitness"], val["eval"]["other"]
    F[i,j] = f_r if f_r>f_b else -f_b


# %%
red_ratings, blue_ratings = asymmetric_elo(F)

# %%
n = 5
red_elites = [reds[i] for i in np.argsort(-red_ratings)[:n]]
blue_elites = [blues[i] for i in np.argsort(-blue_ratings)[:n]]

# %% [markdown]
# ### best of each weight categories

# %%
red_weights = [np.sum(red['wrestler'] > 0) for red in reds]
blue_weights = [np.sum(blue['wrestler'] > 0) for blue in blues]

# %%
red_categories = np.linspace(np.min(red_weights), np.max(red_weights), 6)
red_indices = [[] for _ in range(5)]
for i, w in enumerate(red_weights):
    for j in range(5):
        if red_categories[j] <= w <= red_categories[j+1]:
            red_indices[j].append(i)
red_champions = [red_indices[i][np.argmax( red_ratings[red_indices[i]])] for i in range(5)]

# %%
blue_categories = np.linspace(np.min(blue_weights), np.max(blue_weights), 6)
blue_indices = [[] for _ in range(5)]
for i, w in enumerate(blue_weights):
    for j in range(5):
        if blue_categories[j] <= w <= blue_categories[j+1]:
            blue_indices[j].append(i)
blue_champions = [blue_indices[i][np.argmax( blue_ratings[blue_indices[i]])] for i in range(5)]


# %%
def compute_tournament_for_video(config, reds, blues):
    tournament = {}
    for i, red_team in enumerate(reds):
        for j, blue_team in enumerate(blues):
            blue_team["generation"] = "red"
            tournament[(i,j)] = {"candidate": {"value":red_team}, "task": {"id": j, "config": blue_team}} 
    batch_size = config["batch_size"]
    for i in tqdm(range(0, len(tournament), batch_size)):
        keys = list(tournament.keys())[i:i+batch_size]
        if len(keys) < batch_size:
            keys += [keys[-1]]*(batch_size-len(keys))
        candidates = [tournament[key]["candidate"] for key in keys]
        tasks = [tournament[key]["task"] for key in keys]
        evaluations = evaluate_wrestlers_for_video(**config["evaluation_config"], resolution=(720, 720), tasks=tasks, candidates=candidates)
        for j, key in enumerate(keys):
            tournament[key]["eval"] = evaluations[j]
    return tournament


# %%
red_elites = [reds[i] for i in red_champions]
blue_elites = [blues[i] for i in blue_champions]

# %%
config = get_config(wrestlers_config, None, None, None)
tournament_video = compute_tournament_for_video(config, red_elites, blue_elites)

# %%
F = np.empty((5,5))
for (i,j), duel in tournament_video.items():
    f_r, f_b = duel["eval"]["fitness"], duel["eval"]["other"]
    F[i,j] = 100 * f_r

# %%
plt.subplots(figsize=(7,6))
bound = max(abs(np.min(F)), abs(np.max(F)))
plt.pcolor(F[:, ::-1].T, cmap=cm.coolwarm, vmin=0, vmax=bound,)
plt.colorbar(label="Red closest to center (%)")
#plt.axis("equal")
#plt.axis("off")
plt.yticks([0.5,1.5,2.5,3.5,4.5], np.array(blue_weights)[blue_champions][::-1])
plt.xticks([0.5,1.5,2.5,3.5,4.5], np.array(red_weights)[red_champions])
plt.ylabel("Blue Weight")
plt.xlabel("Red Weight")
plt.tight_layout()

# %%
W, H = n, n
image_size = 200
timestep = 1
pad = 2
W *= (image_size+2*pad)
H *= (image_size+2*pad)

images = []
for t in range(n_steps//timestep):
    new = Image.new("RGBA", (W, H))
    draw = ImageDraw.Draw(new)
    new.paste(Image.new("RGBA", new.size, "WHITE"), (0,0))
    images.append(new)

# %%
for (i, j), duel in tqdm(tournament_video.items()):
    frames = duel["eval"]["rgb_array"]
    for t in range(n_steps//timestep):
        (x_pos, y_pos) = i, j
        y = y_pos * (image_size+2*pad) 
        x = x_pos * (image_size+2*pad) 
        img = Image.fromarray(frames[t].astype(np.uint8))
        img = img.resize((image_size, image_size))
        images[t].paste(img, (x, y))
        images[t].paste(img, (x+pad, y+pad))

# %%
path = gen_folder_path + f'/weight_classes'
Tmax = 10
images[0].save( path + '.gif', save_all=True, append_images=images[1:], optimize=True, duration=25, loop=0)

videodims = images[0].size
fourcc = cv2.VideoWriter_fourcc(*'avc1')    
video = cv2.VideoWriter(path + ".mp4", fourcc, 1000//25, videodims)
img = Image.new('RGB', videodims, color = 'darkred')
for image in images:
    video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
video.release()


# %% [markdown]
# ### record video

# %%
def compute_tournament_for_video(config, reds, blues):
    tournament = {}
    for i, red_team in enumerate(reds):
        for j, blue_team in enumerate(blues):
            blue_team["generation"] = "red"
            tournament[(i,j)] = {"candidate": {"value":red_team}, "task": {"id": j, "config": blue_team}} 
    batch_size = config["batch_size"]
    for i in tqdm(range(0, len(tournament), batch_size)):
        keys = list(tournament.keys())[i:i+batch_size]
        if len(keys) < batch_size:
            keys += [keys[-1]]*(batch_size-len(keys))
        candidates = [tournament[key]["candidate"] for key in keys]
        tasks = [tournament[key]["task"] for key in keys]
        evaluations = evaluate_wrestlers_for_video(**config["evaluation_config"], resolution=(720, 720), tasks=tasks, candidates=candidates)
        for j, key in enumerate(keys):
            tournament[key]["eval"] = evaluations[j]
    return tournament


# %%
config = get_config(wrestlers_config, None, None, None)
tournament_video = compute_tournament_for_video(config, red_elites, blue_elites)

# %%
F = np.empty((n,n))
for (i,j), duel in tournament_video.items():
    f_r, f_b = duel["eval"]["fitness"], duel["eval"]["other"]
    F[i,j] = f_r if f_r>f_b else -f_b

# %%
plt.subplots(figsize=(7,7))
bound = max(abs(np.min(F)), abs(np.max(F)))
plt.pcolor(F[:, ::-1].T, cmap=cm.coolwarm, vmin=-bound, vmax=bound,)
plt.colorbar(label="Time closest to center (%)")
plt.axis("equal")
plt.axis("off")
plt.tight_layout()

# %%
W, H = n, n
image_size = 200
timestep = 1
pad = 2
W *= (image_size+2*pad)
H *= (image_size+2*pad)

images = []
for t in range(n_steps//timestep):
    new = Image.new("RGBA", (W, H))
    draw = ImageDraw.Draw(new)
    new.paste(Image.new("RGBA", new.size, "WHITE"), (0,0))
    images.append(new)

# %%
for (i, j), duel in tqdm(tournament_video.items()):
    frames = duel["eval"]["rgb_array"]
    for t in range(n_steps//timestep):
        (x_pos, y_pos) = i, j
        y = y_pos * (image_size+2*pad) 
        x = x_pos * (image_size+2*pad) 
        img = Image.fromarray(frames[t].astype(np.uint8))
        img = img.resize((image_size, image_size))
        images[t].paste(img, (x, y))
        images[t].paste(img, (x+pad, y+pad))

# %%
path = gen_folder_path + f'/umap'
Tmax = 10
images[0].save( path + '.gif', save_all=True, append_images=images[1:], optimize=True, duration=25, loop=0)

videodims = images[0].size
fourcc = cv2.VideoWriter_fourcc(*'avc1')    
video = cv2.VideoWriter(path + ".mp4", fourcc, 1000//25, videodims)
img = Image.new('RGB', videodims, color = 'darkred')
for image in images:
    video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
video.release()

# %% [markdown]
# ### Diversity sample

# %%
gen_id = 9
tournament =  utils.load_pickle(gen_folder_path + f"tournament_{gen_id}.pk")
Elites = utils.load_pickle(gen_folder_path + f"elites_{gen_id}.pk")

blues = Elites["blues"]
reds = Elites["reds"]
n_red, n_blue = len(reds), len(blues)

B = []
for (i,j), val in tournament.items():
    B.append( val["eval"]["behavior"])
B = np.array(B)

F = np.zeros((n_red, n_blue))
f = []
for (i,j), val in tournament.items():
    f_r, f_b = val["eval"]["fitness"], val["eval"]["other"]
    f.append(f_r if f_r>f_b else -f_b)
    F[i,j] = f_r if f_r>f_b else -f_b

# %%
kmeans = KMeans(n_clusters=25).fit(preprocessing.normalize(B))   # for normalized vectors euclidian distance is equivalent to cosine distance ||x-y||² = 2 (1 - x.y)
centroids = kmeans.cluster_centers_

# %%
all_B = np.array(B)
all_B /= np.linalg.norm(all_B, axis=1)[:, np.newaxis]
pca = PCA(n_components=2)
pca.fit(all_B)
projection = pca.transform(all_B)

# %%
i = 0
plt.subplots(figsize=(7,7))
plt.axis("off")

plt.title("PCA")
plt.axis("off")
plt.scatter(projection[:, 0], projection[:, 1], s=2, c=f, alpha=1, cmap="coolwarm",)

# %%
tree = cKDTree(centroids)

# %%
distances_to_centroids = np.ones(25) * np.inf
centroids = [None for _ in range(25)]
for (i,j), val in tournament.items():
    b = val["eval"]["behavior"]
    distance, c_id = tree.query(b/jnp.linalg.norm(b), k=1)
    if distance < distances_to_centroids[c_id]:
        distances_to_centroids[c_id] = distance
        centroids[c_id] = (i,j)


# %%
def compute_for_video(config, duels, blues, reds):
    tournament = {}
    for i, j in duels:
        red_team = reds[i]
        blue_team = blues[j]
        blue_team["generation"] = "red"
        tournament[(i,j)] = {"candidate": {"value":red_team}, "task": {"id": j, "config": blue_team}} 
    batch_size = config["batch_size"]
    for i in tqdm(range(0, len(tournament), batch_size)):
        keys = list(tournament.keys())[i:i+batch_size]
        if len(keys) < batch_size:
            keys += [keys[-1]]*(batch_size-len(keys))
        candidates = [tournament[key]["candidate"] for key in keys]
        tasks = [tournament[key]["task"] for key in keys]
        evaluations = evaluate_wrestlers_for_video(**config["evaluation_config"], resolution=(720, 720), tasks=tasks, candidates=candidates)
        for j, key in enumerate(keys):
            tournament[key]["eval"] = evaluations[j]
    return tournament


# %%
config = get_config(wrestlers_config, None, None, None)
video_tournament = compute_for_video(config, centroids, blues, reds)  # checked that it returns the same behavior/fitness 

# %%
n = 5
F = np.empty((n,n))
for i, duel in enumerate(video_tournament.values()):
    f_r, f_b = duel["eval"]["fitness"], duel["eval"]["other"]
    F[i%n,i//n] = f_r if f_r>f_b else -f_b

# %%
plt.subplots(figsize=(7,7))
bound = max(abs(np.min(F)), abs(np.max(F)))
plt.pcolor(100*F[:, ::-1].T, cmap=cm.coolwarm, vmin=-100, vmax=100,)
plt.colorbar(label="Time closest to center (%)")
plt.axis("equal")
plt.axis("off")
plt.tight_layout()

# %%
for i, (key, duel) in tqdm(enumerate(video_tournament.items())):
    print(i%n, i//n, key)

# %%
W, H = n, n
image_size = 200
timestep = 1
pad = 2
W *= (image_size+2*pad)
H *= (image_size+2*pad)

images = []
for t in range(n_steps//timestep):
    new = Image.new("RGBA", (W, H))
    draw = ImageDraw.Draw(new)
    new.paste(Image.new("RGBA", new.size, "WHITE"), (0,0))
    images.append(new)

# %%
for i, duel in tqdm(enumerate(video_tournament.values())):
    frames = duel["eval"]["rgb_array"]
    for t in range(n_steps//timestep):
        (x_pos, y_pos) = i%n, i//n
        y = y_pos * (image_size+2*pad) 
        x = x_pos * (image_size+2*pad) 
        img = Image.fromarray(frames[t].astype(np.uint8))
        img = img.resize((image_size, image_size))
        images[t].paste(img, (x, y))
        images[t].paste(img, (x+pad, y+pad))

# %%
path = gen_folder_path + f'/diversity'
Tmax = 10
images[0].save( path + '.gif', save_all=True, append_images=images[1:], optimize=True, duration=25, loop=0)

videodims = images[0].size
fourcc = cv2.VideoWriter_fourcc(*'avc1')    
video = cv2.VideoWriter(path + ".mp4", fourcc, 1000//25, videodims)
img = Image.new('RGB', videodims, color = 'darkred')
for image in images:
    video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
video.release()

# %% [markdown]
# ### Mass distribution per generation

# %%
from scipy.stats import gaussian_kde

plt.subplots(figsize=(12,12))
plt.axis("off")
x_range = np.linspace(0, 25, 200)

for i, (cmap, side) in enumerate(zip([cm.Blues, cm.Reds], ["blues", "reds"])):
    plt.subplot2grid((2,1), (i, 0))
    for gen_id in range(10):
        gen_folder_path = root_folder +  f"gen_{gen_id}/"
        Elites = utils.load_pickle(gen_folder_path + f"elites_{gen_id}.pk")
    
        weights = [np.sum(blue['wrestler'] > 0) for blue in Elites[side]]
        kde = gaussian_kde(weights)
        density_curve = kde(x_range)
        plt.plot(x_range, density_curve, c=plot.int_to_color(gen_id, 19, -5, cmap=cmap), linewidth=3, label=gen_id )
    
        
    plt.legend(fontsize=10, title="Generation", ncols=4)
    plt.grid(axis="x", alpha=0.5)
    plt.xlabel("Weight (#non empty voxels)")
    plt.ylabel("Density")

# %% [markdown]
# ### Fitness vs weight

# %%
plt.subplots(figsize=(20,8))
plt.axis("off")
for gen_id in range(1, 10):
    gen_folder_path = root_folder +  f"gen_{gen_id}/"
    tournament =  utils.load_pickle(gen_folder_path + f"tournament_{gen_id}.pk")
    Elites = utils.load_pickle(gen_folder_path + f"elites_{gen_id}.pk")

    blues = Elites["blues"]
    reds = Elites["reds"]
    
    red_weights = [np.sum(red['wrestler'] > 0) for red in reds]
    blue_weights = [np.sum(blue['wrestler'] > 0) for blue in blues]
    
    n_red, n_blue = len(reds), len(blues)
    F = np.ones((n_red, n_blue))*0.5
    for (i,j), val in tournament.items():
        f_r, f_b = val["eval"]["fitness"], val["eval"]["other"]
        F[i,j] = f_r 

    
    red_indices = np.argsort(red_weights)
    blue_indices = np.argsort(blue_weights)
    
    bound = max(abs(np.min(F)), abs(np.max(F)))
    plt.subplot2grid((2,5), (gen_id%2, gen_id//2))
    plt.pcolor(F[red_indices][:, blue_indices], vmin=0, vmax=1, cmap="coolwarm")
    #plt.colorbar(label="Time closest to center (%)")
    #plt.axis("equal")
    plt.title(f"{gen_id}")
    plt.tight_layout()
    indices = np.linspace(0, len(blue_indices)-1, 5, dtype=np.int_)
    plt.xticks(indices+0.5, np.array(blue_weights)[blue_indices[indices]], fontsize=8);
    #plt.ylabel("Red weights")
    indices = np.linspace(0, len(red_indices)-1, 5, dtype=np.int_)
    plt.yticks(indices+0.5, np.array(red_weights)[red_indices[indices]], fontsize=8);
    #plt.xlabel("Blue weights")

# %% [markdown]
# ### Best of each generation

# %%
bests = []
for gen_id in tqdm(range(1, 10)):
    gen_folder_path = root_folder +  f"gen_{gen_id}/"
    tournament =  utils.load_pickle(gen_folder_path + f"tournament_{gen_id}.pk")
    Elites = utils.load_pickle(gen_folder_path + f"elites_{gen_id}.pk")
    
    blues = Elites["blues"]
    reds = Elites["reds"]
    
    red_weights = [np.sum(red['wrestler'] > 0) for red in reds]
    blue_weights = [np.sum(blue['wrestler'] > 0) for blue in blues]
    
    n_red, n_blue = len(reds), len(blues)
    F = np.zeros((n_red, n_blue))
    for (i,j), val in tournament.items():
        f_r, f_b = val["eval"]["fitness"], val["eval"]["other"]
        F[i,j] = f_r if f_r>f_b else -f_b
    red_ratings, blue_ratings = asymmetric_elo(F)
    bests.append({"red": reds[np.argmax(red_ratings)], "blue": blues[np.argmax(blue_ratings)]})

# %%
plt.subplots(figsize=(20,8))
plt.axis("off")
for gen_id in range(9):
    plt.subplot2grid((2,5), ((1+gen_id)//5, (1+gen_id)%5))
    plt.imshow(viz_wrestler(bests[gen_id]["red"]["wrestler"]))
    plt.title(gen_id+1)
    plt.axis("off")

# %% [markdown]
# ### All tournaments

# %%

plt.subplots(figsize=(20,8))
plt.axis("off")
   
for gen_id in tqdm(range(1, 10)):
    gen_folder_path = root_folder +  f"gen_{gen_id}/"
    tournament =  utils.load_pickle(gen_folder_path + f"tournament_{gen_id}.pk")
    Elites = utils.load_pickle(gen_folder_path + f"elites_{gen_id}.pk")
    
    blues = Elites["blues"]
    reds = Elites["reds"]
        
    n_red, n_blue = len(reds), len(blues)
    F = np.ones((n_red, n_blue))*0.5
    for (i,j), val in tournament.items():
        f_r, f_b = val["eval"]["fitness"], val["eval"]["other"]
        F[i,j] = f_r
        
    red_indices = jnp.argsort(jnp.mean(F, axis=1))
    blue_indices = jnp.argsort(-jnp.mean(F, axis=0))

    plt.subplot2grid((2,5), ((gen_id)%2, (gen_id)//2))
    plt.pcolor(F[red_indices][:, blue_indices], vmin=0, vmax=1, cmap="coolwarm")
    plt.title(f"{gen_id}")
    plt.axis("off")

# %% [markdown]
# ### inter-gen tournament

# %%
n_gen = 10
Elites = [utils.load_pickle(root_folder + f"/gen_{gen_id}/elites_{gen_id}.pk") for gen_id in range(n_gen)]

blues, reds = [], []
for gen_id in [i for i in range(1, n_gen, 2)]:
    for blue in Elites[gen_id]["blues"]:
        blues.append(blue)
for gen_id in [i for i in range(0, n_gen, 2)]:
    for red in Elites[gen_id]["reds"]:
        reds.append(red)

# compute red generation start (i) and end (j)
i, red_indices = 0, {}
for gen_id in range(0, n_gen, 2):
    n_reds = len(Elites[gen_id]["reds"])
    j = i + n_reds
    red_indices[gen_id] = (i,j)
    i = j 

# compute blue generation start (i) and end (j)
i, blue_indices = 0, {}
for gen_id in range(1, n_gen, 2):
    n_blues = len(Elites[gen_id]["blues"])
    j = i + n_blues
    blue_indices[gen_id] = (i,j)
    i = j 

n_red, n_blue = len(reds), len(blues)
F_red = np.ones((n_red, n_blue))*0.5
for blue_gen_id, (blue_i, blue_j) in blue_indices.items():
    for red_gen_id, (red_i, red_j) in red_indices.items():
        mini_tournament = utils.load_pickle(root_folder + f"generational_tournament_{blue_gen_id}_{red_gen_id}.pk")
        print(blue_gen_id, red_gen_id)
        for (i, j), val in mini_tournament.items():
            F_red[i,j] = val["eval"]["fitness"]

# %% [markdown]
# #### tournament plot

# %%
sorted_red_indices = np.concatenate([i+jnp.argsort(jnp.mean(F_red[i:j], axis=1)) for (i,j) in red_indices.values()])
sorted_blue_indices = np.concatenate([i+jnp.argsort(-jnp.mean(F_red[:, i:j], axis=0))  for (i,j) in blue_indices.values()])

# %%
plt.subplots(figsize=(8,8))
plt.pcolor(F_red[sorted_red_indices][:, sorted_blue_indices], vmin=0, vmax=1, cmap=cm.coolwarm)
plt.xticks([(i+j)/2 for (i,j) in blue_indices.values()], blue_indices.keys())
plt.yticks([(i+j)/2 for (i,j) in red_indices.values()], red_indices.keys())
plt.xlabel("Blues")
plt.ylabel("Reds")

# %% [markdown]
# #### elo per gen

# %%
Red_ratings, Blue_ratings = asymmetric_elo(F_red)

# %%
data = [Blue_ratings[blue_i:blue_j] for (blue_i, blue_j) in blue_indices.values()]
names = [str(i) for i in range(1, n_gen, 2)]
plot.plot_boxplot(data, names, cmap=cm.Blues, figsize=(16,5));
data = [Red_ratings[i:j] for (i, j) in red_indices.values()]
names = [str(i) for i in range(0, n_gen, 2)]
plot.plot_boxplot(data, names, cmap=cm.Reds, figsize=(16,5));

# %% [markdown]
# #### adversarial stepping stones 

# %%
A = {}
B = {}
for gen_id in range(9):
    if gen_id % 2 == 0: # red
        red_i, red_j =  red_indices[gen_id]
        blue_i, blue_j = blue_indices[gen_id+1]
        red, blue = np.where(F_red[red_i:red_j, blue_i:blue_j]<0.5)
        old, new = red_i + red, blue_i + blue
    else:
        red_i, red_j =  red_indices[gen_id+1]
        blue_i, blue_j = blue_indices[gen_id]
        red, blue = np.where(F_red[red_i:red_j, blue_i:blue_j]>0.5)
        new, old = red_i + red, blue_i + blue
    for (i,j) in zip(new, old):
        new_key = (('b' if gen_id % 2 == 0 else 'r'), i)
        old_key = (('r' if gen_id % 2 == 0 else 'b'), j)
        if old_key not in A:
            A[old_key] = []
        if new_key not in B:
            B[new_key] = []
        A[old_key].append(new_key)
        B[new_key].append(old_key)

# %%
gens = []
for gen_id in range(10):
    gen = []
    if gen_id % 2 == 0: # red
        for i in range(*red_indices[gen_id]):
            gen.append(("r", i))
    else:
        for i in range(*blue_indices[gen_id]):
            gen.append(("b", i))
    gens.append(gen)

# %%
for gen_id, gen in enumerate(gens[1:]):
    for u in gen:
        if u in B:
            plt.scatter([gen_id], [gen.index(u)], s=len(B[u]), color=plot.int_to_color(len(B[u]), 50, cmap=cm.viridis))

# %%
vertex = []
for gen_id in range(10):
    if gen_id % 2 == 0: # red
        for i in range(*red_indices[gen_id]):
            vertex.append(("r", i))
    else:
        for i in range(*blue_indices[gen_id]):
            vertex.append(("b", i))

# %%
distance = {('r', i): 0 for i in range(*red_indices[0])}
for key in vertex:
    if key not in distance:
        distance[key] = -np.inf
parent = {v: None for v in vertex}

# %%
for u in vertex:
    if u in A:
        for v in A[u]:
            if distance[u] + len(B[v]) > distance[v]:
                distance[v] = distance[u] + len(B[v]) 
                parent[v] = u

# %%
end = list(distance.keys())[np.argmax(list(distance.values()))]

# %%
ancestors = []
current = end
while current is not None:
    ancestors.append(current)
    current = parent[current]
ancestors = ancestors[::-1]

# %%
duels = []
for gen_id in range(9):
    old, new = ancestors[gen_id], ancestors[gen_id+1]
    if gen_id % 2 == 0:
        duels.append((old[1], new[1]))
        print(gen_id, 1-F_red[old[1], new[1]])
    else:
        duels.append((new[1], old[1]))
        print(gen_id, F_red[new[1], old[1]])


# %%
def compute_for_video(config, duels, blues, reds):
    tournament = {}
    for i, j in duels:
        red_team = reds[i]
        blue_team = blues[j]
        blue_team["generation"] = "red"
        tournament[(i,j)] = {"candidate": {"value":red_team}, "task": {"id": j, "config": blue_team}} 
    batch_size = config["batch_size"]
    for i in tqdm(range(0, len(tournament), batch_size)):
        keys = list(tournament.keys())[i:i+batch_size]
        if len(keys) < batch_size:
            keys += [keys[-1]]*(batch_size-len(keys))
        candidates = [tournament[key]["candidate"] for key in keys]
        tasks = [tournament[key]["task"] for key in keys]
        evaluations = evaluate_wrestlers_for_video(**config["evaluation_config"], resolution=(720, 720), tasks=tasks, candidates=candidates)
        for j, key in enumerate(keys):
            tournament[key]["eval"] = evaluations[j]
    return tournament


# %%
config = get_config(wrestlers_config, None, None, None)
video_tournament = compute_for_video(config, duels, blues, reds)  # checked that it returns the same behavior/fitness 

# %%
n = len(duels)
F = np.empty((n))
for i, duel in enumerate(video_tournament.values()):
    f_r = duel["eval"]["fitness"]
    F[i] = f_r 

# %%
W, H = 5, 2
image_size = 200
timestep = 1
pad = 2
W *= (image_size+2*pad)
H *= (image_size//2+2*pad)

images = []
for t in range(n_steps//timestep):
    new = Image.new("RGBA", (int(W), int(H)))
    draw = ImageDraw.Draw(new)
    new.paste(Image.new("RGBA", new.size, "BLACK"), (0,0))
    images.append(new)
# Try to use a default font, fallback to default if not available
try:
    font = ImageFont.truetype("arial.ttf", 20)  # Adjust size as needed
except:
    font = ImageFont.load_default()
            

for i, duel in tqdm(enumerate(video_tournament.values())):
    frames = duel["eval"]["rgb_array"]
    for t in range(n_steps//timestep):
        (x_pos, y_pos) = 0.5*i, i%2
        y = y_pos * (image_size//2+2*pad) 
        x = x_pos * (image_size+2*pad) 
        if i%2 == 1:
            img = Image.fromarray(frames[t].astype(np.uint8)[:360, :-1])
        else:
            img = Image.fromarray(frames[t].astype(np.uint8)[:360, ::-1][:, 1:])
        img = img.resize((image_size, image_size//2))
        
        
        draw = ImageDraw.Draw(img)
        
        text = f"{i} vs {i+1}" 
        
        # Get text bounding box to calculate size
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate centered position
        text_x = (img.width - text_width) // 2 
        text_y = (text_height) // 2 

        draw.text((text_x, text_y), text, font=font, fill="black")
        
        f = (1-F[i]) if i%2==1 else F[i]
        
        draw.text((5, (text_height)//2), f"{f*100:2.0f}%", font=font, fill="black")
        draw.text((img.width-50, (text_height)//2), f"{(1-f)*100:2.0f}%", font=font, fill="black")
        
        images[t].paste(img, (int(x+pad), int(y+pad)))

# %%
path = root_folder + f'/stepping_stones_{"_".join([str(k[1]) for k in ancestors])}'
Tmax = 10
images[0].save( path + '.gif', save_all=True, append_images=images[1:], optimize=True, duration=25, loop=0)

videodims = images[0].size
fourcc = cv2.VideoWriter_fourcc(*'avc1')    
video = cv2.VideoWriter(path + ".mp4", fourcc, 1000//25, videodims)
img = Image.new('RGB', videodims, color = 'darkred')
for image in images:
    video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
video.release()

# %% [markdown]
# #### Clustering and projection

# %%
robot_list = [r['wrestler'] for r in reds]
robot_matrices = np.array(robot_list)
n_robots = len(robot_list)
red_robot_vectors = robot_matrices.reshape(n_robots, -1)

red_on_hot = np.zeros((n_robots, 25, 7))
for i, robot in enumerate(red_robot_vectors):
    for j, x in enumerate(robot):
        red_on_hot[i,j,int(x)] = 1
red_on_hot = red_on_hot.reshape(n_robots, -1)

robot_list = [b['wrestler'] for b in blues]
robot_matrices = np.array(robot_list)
n_robots = len(robot_list)
blue_robot_vectors = robot_matrices.reshape(n_robots, -1)

blue_on_hot = np.zeros((n_robots, 25, 7))
for i, robot in enumerate(blue_robot_vectors):
    for j, x in enumerate(robot):
        blue_on_hot[i,j,int(x)] = 1
blue_on_hot = blue_on_hot.reshape(n_robots, -1)

# %%
random_state = 42
n_clusters = 5
kmode = KModes(n_clusters=n_clusters, init = "random", n_init = 5, verbose=0)
red_cluster_labels = kmode.fit_predict(red_robot_vectors)
blue_cluster_labels = kmode.fit_predict(blue_robot_vectors)

# %%
Bests = []
Indices = []
for gen_id in range(10):
    if gen_id%2 == 0: 
        i, I = red_indices[gen_id]
        j, J = blue_indices[gen_id+1]
    else:
        i, I = red_indices[gen_id-1]
        j, J = blue_indices[gen_id]

    red_ratings, blue_ratings = asymmetric_elo(F_red[i:I, j:J])
    idx = {}
    bests = {}
    for c in range(n_clusters):
        if gen_id%2 ==0:
            valid = np.where(red_cluster_labels[i:I] == c)[0]
            if len(valid)>0:
                idx[c] =  i + valid[np.argmax(red_ratings[valid])]
                bests[c] = reds[idx[c]]["wrestler"]
        else:
            valid = np.where(blue_cluster_labels[j:J] == c)[0]
            if len(valid)>0:
                idx[c] = j + valid[np.argmax(blue_ratings[valid])]
                bests[c] = blues[idx[c]]["wrestler"]
    Bests.append(bests)
    Indices.append(idx)

# %%
plt.subplots(figsize=(10, 10))
plt.axis("off")
for gen_id in range(0,10,2):
    for c in range(n_clusters):
        plt.subplot2grid((n_clusters,5), (c, gen_id//2))
        if c in Bests[gen_id]:
            plt.imshow(viz_wrestler(Bests[gen_id][c]))
        if gen_id == 0:
            plt.ylabel(f"cluster: {c}", fontsize=8)
        if c == 0:
            plt.title(f"gen: {gen_id}")
        plt.xticks([])
        plt.yticks([])

# %%
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

side = "Blue"
s = int(side=="Blue")

fig, ax = plt.subplots()

y_min, y_max = np.inf, -np.inf

score_per_gen = [[] for _ in range(5)]
active_gen = []
for c in range(n_clusters):
    elo_score = [Red_ratings[indices[c]] if side == "Red" else Blue_ratings[indices[c]] for indices in Indices[s::2] if c in indices]
    gen_ids = [gen_id for gen_id in range(5) if c in Indices[s::2][gen_id]]
    active_gen.append(gen_ids)
    y_min = min(y_min, np.min(elo_score))
    y_max = max(y_max, np.max(elo_score))
    for gen_id, y in zip(gen_ids, elo_score):
        score_per_gen[gen_id].append(y)  

for c in range(n_clusters):
    elo_score = [Red_ratings[indices[c]] if side == "Red" else Blue_ratings[indices[c]] for indices in Indices[s::2] if c in indices]
    plt.plot(active_gen[c], elo_score, label=c, lw=3)
    plt.scatter(active_gen[c], elo_score, s=100)
    x_id = 0
    for x, gen_id in enumerate(range(s,10,2)):
        scores = list(np.copy(score_per_gen[x]))
        scores.sort()
        pad = (y_max-y_min)/(2*len(scores))
        if c in Bests[gen_id]:
            imagebox = OffsetImage(viz_wrestler(Bests[gen_id][c]), zoom=13) # zoom controls size
            color = plot.int_to_color(c, vmax=10, cmap="tab10")
            ab = AnnotationBbox(imagebox, (x, elo_score[x_id]), alpha=0.5, frameon=True,   
                                xybox=(x+0.5, y_min+(2*scores.index(elo_score[x_id])+1)*pad),
                                arrowprops=dict(arrowstyle="-", color=color, zorder=1),
                                bboxprops=dict(boxstyle="round,pad=0.5", 
                                               facecolor='white',    # background color
                                               edgecolor=color,          # frame color
                                               linewidth=2),
                               zorder=10)
            ax.add_artist(ab)
            x_id += 1
#plt.legend(title="Cluster")
plt.ylabel("Intergenerational Elo Score")
plt.xlabel(f"{side} Generations")
plt.xticks([0,1,2,3,4], [i for i in range(s,10,2)]);

# %%
reducer = umap.UMAP(
    n_neighbors=30,        # Balance between local and global structure
    min_dist=0.1,          # Minimum distance between points in embedding
    n_components=2,        # 2D output
    metric='manhattan',    # Distance metric (could also try 'cosine', 'manhattan')
    random_state=42
)species definition

red_embedding = reducer.fit_transform(red_on_hot)
blue_embedding = reducer.fit_transform(blue_on_hot)

# %%
plt.figure(figsize=(10, 5))
plt.axis("off")

for i, (title, embedding, cluster_labels) in enumerate(zip(["Red", "Blue"], [red_embedding, blue_embedding], [red_cluster_labels, blue_cluster_labels])):
    plt.subplot2grid((1,2), (0, i))
    plt.axis("off")
    plt.title(title)
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], vmax=10,
                         c=cluster_labels, cmap='tab10', alpha=0.7, s=50)

    for gen_id in range(5):
        for idx in Indices[gen_id*2+i].values():
            plt.text(embedding[idx, 0], embedding[idx, 1], str(gen_id*2+i), size=10, ha="center", va="center")
            #plt.scatter(embedding[idx, 0], embedding[idx, 1], color="black", marker="+")

    
#plt.colorbar(scatter, label='Cluster')

# %% [markdown]
# #### cluster vs clusters

# %%
reducer = umap.UMAP(
    n_neighbors=30,        # Balance between local and global structure
    min_dist=0.1,          # Minimum distance between points in embedding
    n_components=1,        # 2D output
    metric='manhattan',    # Distance metric (could also try 'cosine', 'manhattan')
    random_state=42
)

red_embedding = reducer.fit_transform(red_on_hot)
blue_embedding = reducer.fit_transform(blue_on_hot)

# %%
mean_fitness = {}
for red_c in range(n_clusters):
    red_indices = np.where(red_cluster_labels == red_c)[0]
    for blue_c in range(n_clusters):
        blue_indices = np.where(blue_cluster_labels == blue_c)[0]
        mean_fitness[red_c, blue_c] = np.mean(F_red[red_indices][:, blue_indices])

# %%
red_E, blue_E = [], []
for c in range(n_clusters):
    red_indices = np.where(red_cluster_labels == c)[0]
    red_E.append(np.median(red_embedding[red_indices]))
    blue_indices = np.where(blue_cluster_labels == c)[0]
    blue_E.append(np.median(blue_embedding[blue_indices]))

# %%
F_cluster = np.empty((n_clusters, n_clusters))
for red_c in range(n_clusters):
    i = list(np.argsort(red_E)).index(red_c)
    for blue_c in range(n_clusters):
        j = list(np.argsort(blue_E)).index(blue_c)
        F_cluster[i,j] = mean_fitness[red_c, blue_c]

# %%
red_matrices = np.array([r['wrestler'] for r in reds])
blue_matrices = np.array([r['wrestler'] for r in blues])

# %%
plt.subplots(figsize=(11,11))
plt.axis("off")
for c in range(n_clusters):
    i = list(np.argsort(red_E)).index(c) 
    plt.subplot2grid((11,11), (i, 0))
    plt.imshow(viz_wrestler(np.array(np.median(red_matrices[np.where(red_cluster_labels == c)[0]], axis=0), dtype=int)))
    plt.axis("off")
    
    j = list(np.argsort(blue_E)).index(c) 
    plt.subplot2grid((11,11), (10, 1+j))
    plt.imshow(viz_wrestler(np.array(np.median(blue_matrices[np.where(blue_cluster_labels == c)[0]], axis=0), dtype=int)))
    plt.axis("off")
    

for i in range(n_clusters):
    for j in range(n_clusters):
        plt.subplot2grid((11,11), (i, 1+j))
        plt.pcolor([[F_cluster[i,j]]], vmin=0, vmax=1, cmap=cm.coolwarm)    
        plt.axis("off")


# %% [markdown]
# ### Evolution Tree

# %%
def get_evaluations(gen_id):
    archive_save = utils.load_pickle(root_folder+f"gen_{gen_id}/archive_save.pk")
    evaluations = []
    for batch in archive_save["log"]:
        for ev in batch:
            if ev:
                keys = ["origin", "id", "solution", "is_elite"]
                ev = {key: ev[key] for key in keys if key in ev}
                evaluations.append(ev)
    return evaluations
    
def find_id(solution, evaluations):
    for i, ev in enumerate(evaluations):
        if np.linalg.norm(solution["wrestler"] - ev["solution"]["wrestler"]) == 0:
            return i
    return None

def compute_parents(evaluations):
    parents = {}
    for ev in evaluations:
        if "is_elite" in ev and ev["is_elite"]:
            parents[ev["id"]] = ev["origin"]
    return aprents


# %%
E = []
for i in tqdm(range(10)):
    E.append(get_evaluations(i))

# %%
parents = {}
for gen_id in tqdm(range(10)):
    for ev in E[gen_id]:
        if "is_elite" in ev and ev["is_elite"]:
            if "origin" in ev:
                parents[(gen_id, ev["id"])] = (gen_id, ev["origin"])
            else:
                parents[(gen_id, ev["id"])] = (gen_id-2, find_id(ev["solution"], E[gen_id-2]))
utils.save_pickle(root_folder + "parents.pk", parents)

# %% [markdown]
# #### ancestry of one

# %%
best = blues[np.argmax(Blue_ratings)]

# %%
best_id = find_id(best, E[7])

# %%
gen_id = 7
ancestors = []
gen_id, parent = gen_id, best_id
while not (parent == "random"):
    ancestors.append((gen_id, parent))
    gen_id, origin = parents[(gen_id, parent)]
    if type(origin) == int:
        parent = origin
    elif "p1" in origin:
        parent = origin["p1"][1]
    else:
        parent = origin
ancestors.append((gen_id, parent))

# %%
plt.subplots(figsize=(16,9))
plt.axis("off")
n = len(ancestors)-1
h, w = 4, 8
for i, (gen_id, ev_id) in enumerate(ancestors):
    if ev_id != "random":
        plt.subplot2grid((h, w), (i//w, i%w))
        plt.imshow(viz_wrestler(E[gen_id][ev_id]["solution"]["wrestler"]))
        plt.title(f"{gen_id} - {ev_id}", fontsize = 12)
        if i != 0:
            plt.ylabel("^")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

# %% [markdown]
# #### Species

# %%
red_wrestlers, blue_wrestlers = {}, {}
for (gen_id, p_id) in parents.keys():
    if gen_id % 2 == 0:
        red_wrestlers[gen_id, p_id] = E[gen_id][p_id]["solution"]["wrestler"]
    else:
        blue_wrestlers[gen_id, p_id] = E[gen_id][p_id]["solution"]["wrestler"]

# %%
robot_matrices = np.array(list(red_wrestlers.values()))
n_robots = len(red_wrestlers)
red_robot_vectors = robot_matrices.reshape(n_robots, -1)

robot_matrices = np.array(list(blue_wrestlers.values()))
n_robots = len(blue_wrestlers)
blue_robot_vectors = robot_matrices.reshape(n_robots, -1)

# %%
n_clusters = 20 
kmode = KModes(n_clusters=n_clusters, init = "random", n_init = 5, verbose=0)

red_cluster_labels = kmode.fit_predict(red_robot_vectors)
blue_cluster_labels = kmode.fit_predict(blue_robot_vectors)

# %%
for c in range(n_clusters):
    print(c, set(np.array(list(red_wrestlers.keys()))[:, 0][np.where(red_cluster_labels == c)[0]]))

# %%
h = 4
w = 5

for c in range(n_clusters):
    robot_matrices = np.array(list(red_wrestlers.values()))[np.where(red_cluster_labels == c)[0]]
    plt.subplot2grid((h, w), (c//w, c%w))
    plt.imshow(viz_wrestler(np.median(robot_matrices, axis=0)))
    plt.axis("off")
    #plt.title(c)

# %%

# %%
reducer = umap.UMAP(
    n_neighbors=30,        # Balance between local and global structure
    min_dist=0.1,          # Minimum distance between points in embedding
    n_components=2,        # 2D output
    metric='manhattan',    # Distance metric (could also try 'cosine', 'manhattan')
    random_state=42
)
red_embedding = reducer.fit_transform(red_robot_vectors)
blue_embedding = reducer.fit_transform(blue_robot_vectors)

# %%

# %% [markdown]
# ### Are there any circle?

# %%
n_gen = 10
Elites = [utils.load_pickle(root_folder + f"/gen_{gen_id}/elites_{gen_id}.pk") for gen_id in range(n_gen)]

blues, reds = [], []
for gen_id in [i for i in range(1, n_gen, 2)]:
    for blue in Elites[gen_id]["blues"]:
        blues.append(blue)
for gen_id in [i for i in range(0, n_gen, 2)]:
    for red in Elites[gen_id]["reds"]:
        reds.append(red)

# compute red generation start (i) and end (j)
i, red_indices = 0, {}
for gen_id in range(0, n_gen, 2):
    n_reds = len(Elites[gen_id]["reds"])
    j = i + n_reds
    red_indices[gen_id] = (i,j)
    i = j 

# compute blue generation start (i) and end (j)
i, blue_indices = 0, {}
for gen_id in range(1, n_gen, 2):
    n_blues = len(Elites[gen_id]["blues"])
    j = i + n_blues
    blue_indices[gen_id] = (i,j)
    i = j 

n_red, n_blue = len(reds), len(blues)
F_red = np.ones((n_red, n_blue))*0.5
for blue_gen_id, (blue_i, blue_j) in blue_indices.items():
    for red_gen_id, (red_i, red_j) in red_indices.items():
        mini_tournament = utils.load_pickle(root_folder + f"generational_tournament_{blue_gen_id}_{red_gen_id}.pk")
        for (i, j), val in mini_tournament.items():
            F_red[i,j] = val["eval"]["fitness"]

# %%
red_win = {}
blue_win = {}
for red_gen_id in range(0,10,2):
    for blue_gen_id in range(1,10,2):
        red_i, red_j =  red_indices[red_gen_id]
        blue_i, blue_j = blue_indices[blue_gen_id]
        # red 
        red, blue = np.where(F_red[red_i:red_j, blue_i:blue_j]>0.5)
        red, blue = red_i + red, blue_i + blue
        for (r, b) in zip(red, blue):
            if r not in red_win:
                red_win[r] = []
            red_win[r].append(b)
        # blue 
        red, blue = np.where(F_red[red_i:red_j, blue_i:blue_j]<0.5)
        red, blue = red_i + red, blue_i + blue
        for (r, b) in zip(red, blue):
            if b not in blue_win:
                blue_win[b] = []
            blue_win[b].append(r)

# %%
graph = {}
for red, adj in red_win.items():
    graph[('r', red)] = [('b', blue) for blue in adj]
for blue, adj in blue_win.items():
    graph[('b', blue)] = [('r', red) for red in adj]


# %%
def find_all_cycles(graph):
    """
    Find all cycles in a directed graph.
    
    Returns:
        list: List of cycles, where each cycle is a list of nodes
    """
    all_cycles = []
    color = {}
    path = []
    
    def dfs(node):
        if node in color:
            if color[node] == 1:  # Gray node - back edge found (cycle)
                # Find where the cycle starts in current path
                try:
                    cycle_start_idx = path.index(node)
                    cycle = path[cycle_start_idx:] + [node]
                    # Normalize cycle to start with smallest node to avoid duplicates
                    min_idx = cycle[:-1].index(min(cycle[:-1]))  # Exclude last node (duplicate)
                    normalized_cycle = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
                    
                    # Check if this cycle is already found
                    if normalized_cycle not in all_cycles:
                        all_cycles.append(normalized_cycle)
                    return False  # Continue searching for more cycles
                except ValueError:
                    return False
            return False  # Black node - already processed
        
        color[node] = 1  # Mark as visiting (gray)
        path.append(node)
        
        # Visit all neighbors
        for neighbor in graph.get(node, []):
            dfs(neighbor)
        
        path.pop()  # Remove from path when backtracking
        color[node] = 2  # Mark as visited (black)
        return False
    
    # Check all nodes (handles disconnected components)
    for node in tqdm(graph):
        if node not in color:
            dfs(node)
    
    return all_cycles


# %%
C= find_all_cycles(graph)

# %%
n_gens = []
indices = {"r": red_indices, "b": blue_indices}
for cycle in C:
    gens = []
    for (side, pid) in cycle:
        for gen_id, (i, j) in indices[side].items():
            if i <= pid < j:
                gens.append(gen_id)
                break
    n_gens.append(gens)

# %%
c_id = np.argmax([len(set(gens)) for gens in n_gens])
c = C[c_id]
g_min = n_gens[c_id]
for (gens, cycle) in zip(n_gens, C):
    if len(set(gens)) == 8 and len(cycle) < len(c):
        c = cycle
        g_min = gens

# %%

# %%
duels = []
for i in range(len(c)-1):
    x_side, x = c[i]
    y_side, y = c[i+1]
    if x_side == "r":
        duels.append((x, y))
    else:
        duels.append((y, x))

# %%
config = get_config(wrestlers_config, None, None, None)
video_tournament = compute_for_video(config, duels, blues, reds)  # checked that it returns the same behavior/fitness 

# %%
n = len(duels)
F = np.empty((n))
for i, duel in enumerate(video_tournament.values()):
    f_r = duel["eval"]["fitness"]
    F[i] = f_r 

# %%
W, H = 5.5, 6.5
image_size = 200
timestep = 1
pad = 2
W *= (image_size+2*pad)
H *= (image_size//2+2*pad)

w = W - image_size-4
h = H - image_size//2 - 4

images = []
for t in range(n_steps//timestep):
    new = Image.new("RGBA", (int(W), int(H)))
    draw = ImageDraw.Draw(new)
    new.paste(Image.new("RGBA", new.size, "WHITE"), (0,0))
    images.append(new)
# Try to use a default font, fallback to default if not available
try:
    font = ImageFont.truetype("arial.ttf", 20)  # Adjust size as needed
except:
    font = ImageFont.load_default()
            

for i, duel in tqdm(enumerate(video_tournament.values())):
    frames = duel["eval"]["rgb_array"]
    for t in range(n_steps//timestep):
        (x, y) = 0.5*w*np.cos(2*np.pi*i/n)+w/2, 0.5 * h * np.sin(2*np.pi*i/n)+ h/2

        if i%2 == 0:
            img = Image.fromarray(frames[t].astype(np.uint8)[:360, :-1])
        else:
            img = Image.fromarray(frames[t].astype(np.uint8)[:360, ::-1][:, 1:])
        img = img.resize((image_size, image_size//2))
                
        draw = ImageDraw.Draw(img)
        
        text = f"{g_min[i]} vs {g_min[i+1]}" 
        
        # Get text bounding box to calculate size
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate centered position
        text_x = (img.width - text_width) // 2 
        text_y = (text_height) // 2 

        draw.text((text_x, text_y), text, font=font, fill="black")
        
        f = (1-F[i]) if i%2==1 else F[i]
        
        draw.text((5, (text_height)//2), f"{f*100:2.0f}%", font=font, fill="black")
        draw.text((img.width-50, (text_height)//2), f"{(1-f)*100:2.0f}%", font=font, fill="black")

        images[t].paste(Image.new("RGBA", (img.size[0]+4, img.size[1]+4), "BLACK"), (int(x+pad-2), int(y+pad-2)))
        images[t].paste(img, (int(x+pad), int(y+pad)))

# %%
path = root_folder + f'/cycle_{"_".join([str(k[1]) for k in duels])}'
Tmax = 10
images[0].save( path + '.gif', save_all=True, append_images=images[1:], optimize=True, duration=25, loop=0)

videodims = images[0].size
fourcc = cv2.VideoWriter_fourcc(*'avc1')    
video = cv2.VideoWriter(path + ".mp4", fourcc, 1000//25, videodims)
img = Image.new('RGB', videodims, color = 'darkred')
for image in images:
    video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
video.release()

# %% [markdown]
# ## Misc: Tournament fitness array F + generational indices

# %%
folders = []
for variant, replications in Folders.items():
    for folder in replications:
        folders.append((variant, folder))

# %% [raw]
# for variant, folder in tqdm(folders):
#     path = os.path.join(root_folder, variant, folder)
#     if not os.path.exists(path + "/F.pk"):
#         tournament = utils.load_pickle(path + "/generational_tournament.pk")
#         Elites = [utils.load_pickle(path + f"/gen_{gen_id}/elites_{gen_id}.pk") for gen_id in range(n_gen)]
#         blues, reds = [], []
#         for gen_id in [i for i in range(0, n_gen, 2)]:
#             for blue in Elites[gen_id]["blues"]:
#                 blues.append(blue)
#             for red in Elites[gen_id]["reds"]:
#                 reds.append(red)
#         n_red, n_blue = len(reds), len(blues)
#         F = np.zeros((n_red, n_blue))
#         for (i,j), val in tournament.items():
#             red_f = val["eval"]["fitness"][0]
#             blue_f = 1-val["eval"]["other"][0]
#             F[i,j] = 0.5*(red_f+1) if red_f > blue_f else 0.5*(1-blue_f)
#         utils.save_pickle(path + "/F.pk", F)
#         # compute red generation start (i) and end (j)
#         i, red_indices = 0, []
#         for gen_id in range(0, n_gen, 2):
#             n_reds = len(Elites[gen_id]["reds"])
#             j = i + n_reds
#             red_indices += [(i,j)]
#             i = j 
#         utils.save_pickle(path + "/red_indices.pk", red_indices)
#         # compute blue generation start (i) and end (j)
#         i, blue_indices = 0, []
#         for gen_id in range(0, n_gen, 2):
#             n_blues = len(Elites[gen_id]["blues"])
#             j = i + n_blues
#             blue_indices += [(i,j)]
#             i = j 
#         utils.save_pickle(path + "/blue_indices.pk", blue_indices) 

# %%
Red_indices, Blue_indices = {}, {}
for variant, folder in folders:
    path = os.path.join(root_folder, variant, folder)
    key = (variant, folder)
    Red_indices[key] = utils.load_pickle(path + "/red_indices.pk")
    Blue_indices[key] = utils.load_pickle(path + "/blue_indices.pk")
