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
from PIL import Image, ImageDraw 
import umap.umap_ as umap  # pip install umap-learn
from scipy.spatial import cKDTree
from sklearn import preprocessing
from matplotlib.figure import Figure
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from PIL import ImageFont

# %%
import sys
sys.path.append("Misc") 
import utils
from plot import *
import plot
from grid_plot import * 
sys.path.append("Parabellum")  # import utils from parent folder 
from BT_tree import BT, txt2tree
from eval_bi_bts import eval_bi_bts_factory
import grammar
from GAME import *

# %%
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)  # for the video running while JAX is still in use (doesn't like it)

# %%
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


# %% [markdown]
# # Main Figures 

# %% [markdown]
# ## Paths 

# %%
root_folder = ""  # root path for the replications containing six replications folders: full/no_bootstrap/no_bias/handcrafted_behavior/diversity_only/quality_only
figure_folder = ""  # path to save the figures 
video_folder = ""  # path to save the videos
saved_computation_path = "" 

# name of the replication folders. Ex: one full path to the replication = os.path.join(root_folder, "full", replication_folder_name) 
# the following script for plotting expect more than one replication per variant
Folders = {  
    "full": [],
    "no_bootstrap": [],
    "no_bias": [],
    "handcrafted_behavior": [],
    "diversity_only": [],
    "quality_only": [],
}

# %%
blue_color = "#2070b4"
red_color = "#ca171c" 

# %%
line_styles = ["-", (0, (1, 1)),   (0, (5, 1)),  (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5)),  (0, (5, 5)), ]
colors = cm.rainbow
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]
labels = {
    "full": "GAME-MO",
    "no_bootstrap": "GAME-MO (no bootstrap)", 
    "no_bias": "GAME-SO", 
    "handcrafted_behavior": "GAME-MO (no VEM)",
    "diversity_only": "Diversity-only", 
    "quality_only": "Quality-only", 
}
pca_names = ["GAME-MO\n", "GAME-MO\n(no bootstrap)", "GAME-SO\n", "GAME-MO\n (no VEM)", "Diversity\nonly", "Quality\nonly"]

# %%
font = {'size'   : 18}
mpl.rc('font', **font)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# %% [markdown]
# ## Misc: Tournament fitness array F + generational indices

# %%
folders = []
for variant, replications in Folders.items():
    for folder in replications:
        folders.append((variant, folder))

# %%
for variant, folder in tqdm(folders):
    path = os.path.join(root_folder, variant, folder)
    if not os.path.exists(path + "/F.pk"):
        tournament = utils.load_pickle(path + "/generational_tournament.pk")
        Elites = [utils.load_pickle(path + f"/gen_{gen_id}/elites_{gen_id}.pk") for gen_id in range(n_gen)]
        blues, reds = [], []
        for gen_id in [i for i in range(0, n_gen, 2)]:
            for blue in Elites[gen_id]["blues"]:
                blues.append(blue)
            for red in Elites[gen_id]["reds"]:
                reds.append(red)
        n_red, n_blue = len(reds), len(blues)
        F = np.zeros((n_red, n_blue))
        for (i,j), val in tournament.items():
            red_f = val["eval"]["fitness"][0]
            blue_f = 1-val["eval"]["other"][0]
            F[i,j] = 0.5*(red_f+1) if red_f > blue_f else 0.5*(1-blue_f)
        utils.save_pickle(path + "/F.pk", F)
        # compute red generation start (i) and end (j)
        i, red_indices = 0, []
        for gen_id in range(0, n_gen, 2):
            n_reds = len(Elites[gen_id]["reds"])
            j = i + n_reds
            red_indices += [(i,j)]
            i = j 
        utils.save_pickle(path + "/red_indices.pk", red_indices)
        # compute blue generation start (i) and end (j)
        i, blue_indices = 0, []
        for gen_id in range(0, n_gen, 2):
            n_blues = len(Elites[gen_id]["blues"])
            j = i + n_blues
            blue_indices += [(i,j)]
            i = j 
        utils.save_pickle(path + "/blue_indices.pk", blue_indices) 

# %%
Red_indices, Blue_indices = {}, {}
for variant, folder in folders:
    path = os.path.join(root_folder, variant, folder)
    key = (variant, folder)
    Red_indices[key] = utils.load_pickle(path + "/red_indices.pk")
    Blue_indices[key] = utils.load_pickle(path + "/blue_indices.pk")

# %% [markdown]
# ## BT complexity

# %% [markdown]
# ### BT size 

# %%
Size, N_leaves = {}, {}
for variant, folder in tqdm(folders):
    key = (variant, folder)
    path = os.path.join(root_folder, variant, folder)
    Elites = [utils.load_pickle(path + f"/gen_{gen_id}/elites_{gen_id}.pk") for gen_id in range(0, n_gen, 2)]
    size, n_leaves = [], []
    for elites in Elites:
        n_leaves.append(np.max([e["bt"].n_leaves() for e in elites["reds"]+elites["blues"]]))
        size.append(np.mean([e["bt"].size for e in elites["reds"]+elites["blues"]]))
    Size[key] = size
    N_leaves[key] = n_leaves


# %% [raw]
# data, names = [], []
# for variant, replications in Folders.items():
#     rep_data = []
#     for folder in replications:
#         key = (variant, folder)
#         rep_data.append(N_leaves[key])
#     data.append(rep_data)
#     names.append(labels[variant])
#
# plt.subplots(figsize=((12,3.5)))
# for i, Y in enumerate(data):
#     median = np.median(Y, axis=0) 
#     maxi = np.max(Y, axis=0) 
#     mini = np.min(Y, axis=0) 
#     color = colors[i] if type(colors) == list else int_to_color(i, len(data), cmap=colors)
#     x = [j for j in range(0, 20, 2)]
#     plt.plot(x, median, lw=3, color=color, ls=line_styles[i], label=names[i])
#     plt.fill_between(x, mini, maxi , color=color, alpha=0.3)
#
# plt.xlabel("Generations", fontsize=fontsize)
# plt.ylabel("Size (# Nodes)", fontsize=fontsize)
# plt.grid(axis="y", alpha=0.5)
# x_ticks = [f"{i}-{i+1}" for i in range(0, 20,2)]
# plt.xticks(x, x_ticks, fontsize=ticks_fontsize)
# plt.legend(fontsize=fontsize, handlelength=3, bbox_to_anchor=(1,1,0, 0))
# plt.tight_layout()

# %% [markdown]
# ### Entropy

# %%
def compute_entropy(counts):
    probabilities = counts/np.sum(counts)
    valid_probs = probabilities[probabilities > 0]
    return -jnp.sum(valid_probs * jnp.log2(valid_probs))


# %%
for variant, folder in tqdm(folders):
    key = (variant, folder)
    path = os.path.join(root_folder, variant, folder)
    if not os.path.exists(path + "/H.pk"):
        tournament = utils.load_pickle(path + "/generational_tournament.pk")
        blue_indices = utils.load_pickle(path + "/blue_indices.pk") 
        red_indices = utils.load_pickle(path + "/red_indices.pk") 
        H = []
        for gen_id in range(10):
            i_red, j_red = red_indices[gen_id]
            i_blue, j_blue = blue_indices[gen_id]
            h = []
            for i in range(i_red, j_red):
                for j in range(i_blue, j_blue):
                    h.append((compute_entropy(tournament[(i,j)]["eval"]["actions_id"]["red"][1]) + compute_entropy(tournament[(i,j)]["eval"]["actions_id"]["blue"][1]))/2)      
            H.append(np.mean(h))
        utils.save_pickle(path + "/H.pk", H) 

# %% [raw]
# data, names = [], []
# for variant, replications in Folders.items():
#     rep_data = []
#     for folder in replications:
#         key = (variant, folder)
#         path = os.path.join(root_folder, variant, folder)
#         H = utils.load_pickle(path + "/H.pk") 
#         rep_data.append(H)
#     data.append(rep_data)
#     names.append(labels[variant])
#
# line_styles = ["-"] * len(data) if line_styles is None else line_styles
# for i, Y in enumerate(data):
#     median = np.median(Y, axis=0) 
#     maxi = np.max(Y, axis=0) 
#     mini = np.min(Y, axis=0) 
#     color = colors[i] if type(colors) == list else int_to_color(i, len(data), cmap=colors)
#     x = [i for i in range(0, 20, 2)]
#     plt.plot(x, median, lw=3, color=color, ls=line_styles[i], label=names[i])
#     plt.fill_between(x, mini, maxi , color=color, alpha=0.3)
#     plt.xlabel("Generations")
#     plt.ylabel("Entropy (bits)")
#     plt.grid(axis="y", alpha=0.5)
#     plt.title("")
#     x_ticks = [f"{i}-{i+1}" for i in range(0,20,2)]
#     plt.xticks(x, x_ticks)
# plt.legend(fontsize=10, handlelength=5)
# plt.title("Complexity")
# plt.tight_layout()

# %% [markdown]
# ## PCA of visual behaviors through generations 

# %%
B = []
F = []
indices = []

for variant, folder in tqdm(folders):  # 5m30 for 11 folders
    key = (variant, folder)
    path = os.path.join(root_folder, variant, folder)
    tournament = utils.load_pickle(path + "/generational_tournament.pk")
    blue_gen, red_gen = 0, 0
    for gen_id in range(10):
        Elites = utils.load_pickle(path + f"/gen_{gen_id*2}/elites_{gen_id*2}.pk")
        n_reds = len(Elites["reds"])
        n_blues = len(Elites["blues"])
        for i in range(n_reds):
            for j in range(n_blues):
                duel = tournament[(red_gen+i, blue_gen+j)]
                B.append(np.array(duel["eval"]["behavior"]))
                red_f = duel["eval"]["fitness"][0]
                blue_f = 1-duel["eval"]["other"][0]
                F.append(100*red_f if red_f > blue_f else -100*blue_f)
                indices.append((key, gen_id, i, j))
        blue_gen += n_blues
        red_gen += n_reds
F_pca = np.array(F)
utils.save_pickle(saved_computation + "indices.pk", indices)
utils.save_pickle(saved_computation + "F_pca.pk", F_pca)

# %%
indices = utils.load_pickle(saved_computation + "indices.pk")
rev_indices = []
for name in folders:
    rev_indices.append([i for i in range(len(indices)) if indices[i][0]==name])

# %%
all_B = np.array(B)
all_B /= np.linalg.norm(all_B, axis=1)[:, np.newaxis]
pca = PCA(n_components=2)
pca.fit(all_B)
projection = pca.transform(all_B)
utils.save_pickle(saved_computation + "projection.pk", projection)

# %%
projection = utils.load_pickle(saved_computation + "projection.pk")
F_pca = utils.load_pickle(saved_computation + "F_pca.pk")

# %% [markdown]
# #### Side by side

# %%
plt.subplots(figsize=(4*6, 4*2.5))
plt.axis("off")
for i, (variant, folder) in enumerate(folders):
    y = list(Folders).index(variant)
    x = Folders[variant].index(folder)
    plt.subplot2grid((3, 6*5+1), (x, 5*y), colspan=5)
    
    scatter = plt.scatter(projection[rev_indices[i], 0], projection[rev_indices[i], 1], s=2, c=F_pca[rev_indices[i]], vmin=-100, vmax=100, alpha=0.2, cmap="coolwarm",)
    #plt.xlabel(f"PCA 0 ({100*pca.explained_variance_ratio_[0]:2.1f}%)")
    #plt.ylabel(f"PCA 1 ({100*pca.explained_variance_ratio_[1]:2.1f}%)")
    plt.xticks([]);
    plt.yticks([]);
    if x == 0:
        plt.title(pca_names[y], fontsize=44, va="center")

ax_colorbar = plt.subplot2grid((3, 6*5+1), (0, 6*5), rowspan=5)
norm = scatter.norm  # Get the normalization from the scatter
cbar = ColorbarBase(ax_colorbar, cmap=scatter.cmap, norm=norm, orientation='vertical')
cbar.set_label('Fitness', fontsize=44)

plt.tight_layout()
plt.savefig(f"{figure_folder}PCAs.png") 


# %% [markdown]
# #### Coverage

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
for i, name in tqdm(enumerate(folders)):
    red_qd_score, coverage, red_cells = compute_qd_score(standardized_projection[rev_indices[i]], F[rev_indices[i]], step_size=0.01, alpha=1., verbose=False)
    blue_qd_score, coverage, blue_cells = compute_qd_score(standardized_projection[rev_indices[i]], 100-np.array(F[rev_indices[i]]), step_size=0.01, alpha=1., verbose=False)
    measures[name] = {"coverage": coverage, "red_qd_score": red_qd_score, "red_cells": red_cells, "blue_qd_score": blue_qd_score, "blue_cells": blue_cells}
utils.save_pickle(saved_computation + "measures.pk", measures)

# %%
measures = utils.load_pickle(saved_computation + "measures.pk")
indices = utils.load_pickle(saved_computation + "indices.pk")
Red_qd_score, Coverage, Blue_qd_score = {}, {}, {}
for name in folders:
    n_gen_0 = 0
    coverage, red_qd_score, blue_qd_score = [], [], []
    for gen_id in range(10):
        n_gen_0 += len([i for i in range(len(indices)) if indices[i][0]==name and indices[i][1]==gen_id])
        coverage.append(measures[name]["coverage"][n_gen_0-1])
        red_qd_score.append(measures[name]["red_qd_score"][n_gen_0-1])
        blue_qd_score.append(measures[name]["blue_qd_score"][n_gen_0-1])
    Coverage[name] = coverage
    Red_qd_score[name] = red_qd_score
    Blue_qd_score[name] = blue_qd_score

# %% [raw]
# data, names = [], []
# for variant, replications in Folders.items():
#     rep_data = []
#     for folder in replications:
#         key = (variant, folder)
#         path = os.path.join(root_folder, variant, folder)
#         rep_data.append(Coverage[key])
#     data.append(rep_data)
#     names.append(labels[variant])
#
# line_styles = ["-"] * len(data) if line_styles is None else line_styles
# for i, Y in enumerate(data):
#     median = np.median(Y, axis=0) 
#     maxi = np.max(Y, axis=0) 
#     mini = np.min(Y, axis=0) 
#     color = colors[i] if type(colors) == list else int_to_color(i, len(data), cmap=colors)
#     x = [i for i in range(0, 20, 2)]
#     plt.plot(x, median, lw=3, color=color, ls=line_styles[i], label=names[i])
#     plt.fill_between(x, mini, maxi , color=color, alpha=0.3)
#     plt.xlabel("Generations")
#     plt.ylabel("Coverage (%)")
#     plt.grid(axis="y", alpha=0.5)
#     x_ticks = [f"{i}-{i+1}" for i in range(0,20,2)]
#     plt.xticks(x, x_ticks)
# plt.legend(fontsize=10, handlelength=5)
# plt.title("Visual Diversity")
# plt.tight_layout()

# %% [markdown]
# #### QD-Scores

# %% [raw]
# data, names = [], []
# for variant, replications in Folders.items():
#     rep_data = []
#     for folder in replications:
#         key = (variant, folder)
#         rep_data.append((np.array(Blue_qd_score[key]) + np.array(Red_qd_score[key]))/200)
#     data.append(rep_data)
#     names.append(labels[variant])
#     
# line_styles = ["-"] * len(data) if line_styles is None else line_styles
# for i, Y in enumerate(data):
#     median = np.median(Y, axis=0) 
#     maxi = np.max(Y, axis=0) 
#     mini = np.min(Y, axis=0) 
#     color = colors[i] if type(colors) == list else int_to_color(i, len(data), cmap=colors)
#     x = [i for i in range(0, 20, 2)]
#     plt.plot(x, median, lw=3, color=color, ls=line_styles[i], label=names[i])
#     plt.fill_between(x, mini, maxi , color=color, alpha=0.3)
#     plt.xlabel("Generations")
#     plt.ylabel("QD-Score (%)")
#     plt.grid(axis="y", alpha=0.5)
#     plt.title("")
#     x_ticks = [f"{i}-{i+1}" for i in range(0,20,2)]
#     plt.xticks(x, x_ticks)
# plt.legend(fontsize=10, handlelength=5)
# plt.title("Quality Diversity")
# plt.tight_layout()

# %% [markdown]
# ## Open-endedness

# %%
Blue_new_behaviors, Red_new_behaviors = {}, {}
for variant, folder in tqdm(folders):
    key = (variant, folder)
    path = path = os.path.join(root_folder, variant, folder)
    F = utils.load_pickle(path + "/F.pk")
    blue_indices = utils.load_pickle(path + "/blue_indices.pk") 
    red_indices = utils.load_pickle(path + "/red_indices.pk") 
    
    blue_ranking = np.argsort(np.argsort(F, axis=0), axis=0).T
    red_ranking = np.argsort(np.argsort(F, axis=1), axis=1)
    
    red_behaviors, blue_behaviors = set(), set()
    red_new_behaviors, blue_new_behaviors = [], []
    
    for red_gen_id, (i_red, j_red) in enumerate(red_indices):
        for red_rank in red_ranking[i_red:j_red]:
            red_behaviors.add(tuple(red_rank))
        red_new_behaviors.append(len(red_behaviors))
    Red_new_behaviors[key] = red_new_behaviors
    
    for blue_gen_id, (i_blue, j_blue) in enumerate(blue_indices):
        for blue_rank in blue_ranking[i_blue:j_blue]:
            blue_behaviors.add(tuple(blue_rank))
        blue_new_behaviors.append(len(blue_behaviors))
    Blue_new_behaviors[key] = blue_new_behaviors

# %% [raw]
# data, names = [], []
# for variant, replications in Folders.items():
#     rep_data = []
#     for folder in replications:
#         key = (variant, folder)
#         diff = np.array([0] + list(np.array(Red_new_behaviors[key])+np.array(Blue_new_behaviors[key])))
#         rep_data.append((diff[1:]-diff[:-1])/2)
#     data.append(rep_data)
#     names.append(labels[variant])
#     
# line_styles = ["-"] * len(data) if line_styles is None else line_styles
# for i, Y in enumerate(data):
#     median = np.median(Y, axis=0) 
#     maxi = np.max(Y, axis=0) 
#     mini = np.min(Y, axis=0) 
#     color = colors[i] if type(colors) == list else int_to_color(i, len(data), cmap=colors)
#     x = [i for i in range(0, 20, 2)]
#     plt.plot(x, median, lw=3, color=color, ls=line_styles[i], label=names[i])
#     plt.fill_between(x, mini, maxi , color=color, alpha=0.3)
#     y = [i*10 for i in range(11)]
#     plt.yticks(y,y)
#     plt.xlabel("Generations")
#     plt.ylabel("Novelty (%)")
#     plt.grid(axis="y", alpha=0.5)
#     plt.title("")
#     x_ticks = [f"{i}-{i+1}" for i in range(0,20,2)]
#     plt.xticks(x, x_ticks)
# plt.legend(fontsize=10, handlelength=5)
# plt.title("Open-endedness")
# plt.tight_layout()

# %% [raw]
# plt.subplots(figsize=(16,9))
# for i, name in enumerate(folders.keys()):
#     blue, red = Blue_new_behaviors[name], Red_new_behaviors[name]
#     diff = np.array([0] + blue)
#     plt.plot(diff[1:]-diff[:-1], c=plot.int_to_color(i, vmax=len(folders)-1), lw=3, ls=":", label=f"Blue {name}")
#     diff = np.array([0] + red)
#     plt.plot(diff[1:]-diff[:-1], c=plot.int_to_color(i, vmax=len(folders)-1), lw=3, ls="--", label=f"Red {name}")
# plt.legend(fontsize=12)
# plt.xticks([i//2 for i in range(0, n_gen, 2)]);
# plt.grid(axis="x", alpha=0.5)
# plt.xlabel("Generation")
# plt.ylabel("# New behaviors")
# plt.title("Adversarial Ranking Based Behaviors")

# %% [markdown]
# ## Figure concatenation

# %%
fig, _ = plt.subplots(figsize=(16, 7))
plt.axis("off")

fontsize = 17.5
ticks_fontsize = 14

##
#  BT SIZE
##
ax = plt.subplot2grid((3, 2), (0,0))

data, names = [], []
for variant, replications in Folders.items():
    rep_data = []
    for folder in replications:
        key = (variant, folder)
        rep_data.append(Size[key])
    data.append(rep_data)
    names.append(labels[variant])
for i, Y in enumerate(data):
    median = np.median(Y, axis=0) 
    maxi = np.max(Y, axis=0) 
    mini = np.min(Y, axis=0) 
    color = colors[i] if type(colors) == list else int_to_color(i, len(data), cmap=colors)
    x = [j for j in range(0, 20, 2)]
    plt.plot(x, median, lw=3, color=color, ls=line_styles[i], label=names[i])
    plt.fill_between(x, mini, maxi , color=color, alpha=0.3)

#plt.xlabel("Generations", fontsize=fontsize)
plt.ylabel("BT Size (# Nodes)", fontsize=fontsize)
plt.grid(axis="y", alpha=0.5)
x_ticks = [f"{i}-{i+1}" for i in range(0,20,2)]
plt.xticks(x, x_ticks, fontsize=ticks_fontsize)
plt.title("(a) Solution Size", fontsize=fontsize, fontweight='bold')
plt.yticks(fontsize=ticks_fontsize)
#plt.legend(fontsize=fontsize, handlelength=3, bbox_to_anchor=(1,1,0, 0))
plt.tight_layout()

##
#  Complexity
##
ax = plt.subplot2grid((3, 2), (0, 1))

data, names = [], []
for variant, replications in Folders.items():
    rep_data = []
    for folder in replications:
        key = (variant, folder)
        path = os.path.join(root_folder, variant, folder)
        H = utils.load_pickle(path + "/H.pk") 
        rep_data.append(H)
    data.append(rep_data)
    names.append(labels[variant])
for i, Y in enumerate(data):
    median = np.median(Y, axis=0) 
    maxi = np.max(Y, axis=0) 
    mini = np.min(Y, axis=0) 
    color = colors[i] if type(colors) == list else int_to_color(i, len(data), cmap=colors)
    x = [i for i in range(0, 20, 2)]
    plt.plot(x, median, lw=3, color=color, ls=line_styles[i], label=names[i])
    plt.fill_between(x, mini, maxi , color=color, alpha=0.3)
#plt.xlabel("Generations", fontsize=fontsize)
plt.ylabel("Entropy (bits)", fontsize=fontsize)
plt.grid(axis="y", alpha=0.5)
plt.title("")
x_ticks = [f"{i}-{i+1}" for i in range(0,20,2)]
plt.xticks(x, x_ticks, fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
#plt.legend(fontsize=10, handlelength=5)
plt.title("(b) Behavioral Complexity", fontsize=fontsize, fontweight='bold')
plt.tight_layout()

##
#  Coverage
##
ax = plt.subplot2grid((3, 2), (1, 0))

data, names = [], []
for variant, replications in Folders.items():
    rep_data = []
    for folder in replications:
        key = (variant, folder)
        path = os.path.join(root_folder, variant, folder)
        rep_data.append(Coverage[key])
    data.append(rep_data)
    names.append(labels[variant])

for i, Y in enumerate(data):
    median = np.median(Y, axis=0) 
    maxi = np.max(Y, axis=0) 
    mini = np.min(Y, axis=0) 
    color = colors[i] if type(colors) == list else int_to_color(i, len(data), cmap=colors)
    x = [i for i in range(0, 20, 2)]
    plt.plot(x, median, lw=3, color=color, ls=line_styles[i], label=names[i])
    plt.fill_between(x, mini, maxi , color=color, alpha=0.3)
#plt.xlabel("Generations", fontsize=fontsize)
plt.ylabel("Coverage (%)", fontsize=fontsize)
plt.grid(axis="y", alpha=0.5)
x_ticks = [f"{i}-{i+1}" for i in range(0,20,2)]
plt.xticks(x, x_ticks, fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
#plt.legend(fontsize=10, handlelength=5)
plt.title("(c) Visual Diversity", fontsize=fontsize, fontweight='bold')
plt.tight_layout()

##
#  QD-score
##
ax = plt.subplot2grid((3, 2), (1, 1))

data, names = [], []
for variant, replications in Folders.items():
    rep_data = []
    for folder in replications:
        key = (variant, folder)
        rep_data.append((np.array(Blue_qd_score[key]) + np.array(Red_qd_score[key]))/200)
    data.append(rep_data)
    names.append(labels[variant])
line_styles = ["-"] * len(data) if line_styles is None else line_styles
for i, Y in enumerate(data):
    median = np.median(Y, axis=0) 
    maxi = np.max(Y, axis=0) 
    mini = np.min(Y, axis=0) 
    color = colors[i] if type(colors) == list else int_to_color(i, len(data), cmap=colors)
    x = [i for i in range(0, 20, 2)]
    plt.plot(x, median, lw=3, color=color, ls=line_styles[i], label=names[i])
    plt.fill_between(x, mini, maxi , color=color, alpha=0.3)
#plt.xlabel("Generations", fontsize=fontsize)
plt.ylabel("QD-Score", fontsize=fontsize)
plt.grid(axis="y", alpha=0.5)
plt.title("")
x_ticks = [f"{i}-{i+1}" for i in range(0,20,2)]
plt.xticks(x, x_ticks, fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
plt.title("(d) Quality and Visual Diversity", fontsize=fontsize, fontweight='bold')
plt.tight_layout()

##
#  Open endedness
##
ax = plt.subplot2grid((3, 2), (2, 0))
data, names = [], []
for variant, replications in Folders.items():
    rep_data = []
    for folder in replications:
        key = (variant, folder)
        diff = np.array([0] + list(np.array(Red_new_behaviors[key])+np.array(Blue_new_behaviors[key])))
        rep_data.append((diff[1:]-diff[:-1])/2)
    data.append(rep_data)
    names.append(labels[variant])
    
line_styles = ["-"] * len(data) if line_styles is None else line_styles
lines = []
for i, Y in enumerate(data):
    median = np.median(Y, axis=0) 
    maxi = np.max(Y, axis=0) 
    mini = np.min(Y, axis=0) 
    color = colors[i] if type(colors) == list else int_to_color(i, len(data), cmap=colors)
    x = [i for i in range(0, 20, 2)]
    line, = plt.plot(x, median, lw=3, color=color, ls=line_styles[i], label=names[i])
    lines.append(line)
    plt.fill_between(x, mini, maxi , color=color, alpha=0.3)
y = [i*10 for i in range(0, 11, 2)]
plt.yticks(y,y)
plt.xlabel("Generations", fontsize=fontsize)
plt.ylabel("Novelty (%)", fontsize=fontsize)
plt.grid(axis="y", alpha=0.5)
plt.title("")
x_ticks = [f"{i}-{i+1}" for i in range(0,20,2)]
plt.xticks(x, x_ticks, fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
#plt.legend(fontsize=10, handlelength=5, bbox_to_anchor=(1,1,0,0))
plt.title("(e) Open-endedness", fontsize=fontsize, fontweight='bold')
plt.tight_layout()

##
#  legends
##
ax_legend = plt.subplot2grid((3, 2), (2, 1))
ax_legend.axis('off')

# Add the legend to this subplot
ax_legend.legend(handles=lines, loc='center', frameon=False, fontsize=fontsize, handlelength=4)

# Adjust layout
plt.subplots_adjust(wspace=0., hspace=0)
plt.tight_layout()
utils.savefig("comparisons", figure_folder, timestamp=False)

# %% [markdown]
# # Tournament between variants/baselines/method

# %% [markdown]
# ## Compute intra-variants ELO to select the best BTs

# %%
Red_ratings, Blue_ratings = {}, {}
for variant, folder in tqdm(folders):
    name = (variant, folder)
    path = os.path.join(root_folder, variant, folder)
    F = utils.load_pickle(path + "/F.pk")
    red_ratings, blue_ratings = asymmetric_elo(F)
    Red_ratings[name] = red_ratings
    Blue_ratings[name] = blue_ratings

# %%
n_elites = 10

# %%
Red_Elites, Blue_Elites = [], []
Red_Origines, Blue_Origines = [], []

for variant, folder in tqdm(folders):
    name = (variant, folder)
    path = os.path.join(root_folder, variant, folder)
    all_red_Elites, all_blue_Elites = [], []
    for gen_id in range(0, n_gen, 2):
        gen = utils.load_pickle(path + f"/gen_{gen_id}/elites_{gen_id}.pk")
        for elite in gen["reds"]:
            all_red_Elites.append(elite)
        for elite in gen["blues"]:
            all_blue_Elites.append(elite)
    assert len(all_red_Elites) == len(Red_ratings[name])
    assert len(all_blue_Elites) == len(Blue_ratings[name])
    red_Elites, blue_Elites = [], []
    red_Elites_txt, blue_Elites_txt = [], []
    red_Elites_origin, blue_Elites_origin = [], []
    for i in np.argsort(Red_ratings[name])[::-1]:
        bt = all_red_Elites[i]
        bt_txt = bt["bt"].to_txt()
        if bt_txt not in red_Elites_txt:
            red_Elites_txt.append(bt_txt)
            red_Elites.append(bt)
            red_Elites_origin.append(name +(i,))
        if len(red_Elites) == n_elites:
            break
    for i in np.argsort(Blue_ratings[name])[::-1]:
        bt = all_blue_Elites[i]
        bt_txt = bt["bt"].to_txt()
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

# %% [markdown]
# ### compute tournament

# %%
tournaments_path = saved_computation + "comparison_tournaments/"
utils.save_pickle(tournaments_path + f'{18}_{10}_tournament_Folders.pk', (Folders, folders))

# %% [raw]
# config = get_config(parabellum_config, 0, None, n_cells=None, n_groups=n_groups, use_embedding=True)
# comparison_tournament = compute_tournament(config, Red_Elites, Blue_Elites)
# utils.save_pickle(tournaments_path + f'{18}_{10}_comparison_tournament.pk', comparison_tournament)

# %%
comparison_tournament = utils.load_pickle(tournaments_path + f'{18}_{10}_comparison_tournament.pk')
(tournament_Folders, tournament_folders) = utils.load_pickle(tournaments_path + f'{18}_{10}_tournament_Folders.pk')

# %% [markdown]
# ### ELO

# %%
n_red, n_blue = len(Red_Elites), len(Blue_Elites)
F = np.zeros((n_red, n_blue))
for (i,j), val in comparison_tournament.items():
    red_f = val["eval"]["fitness"][0]
    blue_f = 1-val["eval"]["other"][0]
    F[i,j] = 0.5*(red_f+1) if red_f > blue_f else 0.5*(1-blue_f)

# %%
red_ratings, blue_ratings = asymmetric_elo(F)

# %%
red_data = [[] for _ in range(6)] 
blue_data = [[] for _ in range(6)] 
both_data = []

for i, (variant, folder) in enumerate(tournament_folders):
    name = (variant, folder)
    j = list(tournament_Folders.keys()).index(variant)
    red_data[j].append(red_ratings[i*n_elites:(i+1)*n_elites])
    blue_data[j].append(blue_ratings[i*n_elites:(i+1)*n_elites])

# %%
for i in range(6):
    red_data[i] = np.concatenate(red_data[i])
    blue_data[i] = np.concatenate(blue_data[i])
    both_data.append(np.concatenate([red_data[i], blue_data[i]]))

# %%
all_red = np.concatenate(red_data)
all_red = list(np.sort(all_red))
all_blue = np.concatenate(blue_data)
all_blue = list(np.sort(all_blue))

for i in range(6):
    red_data[i] = [all_red.index(x) for x in red_data[i]]
    blue_data[i] = [all_blue.index(x) for x in blue_data[i]]
    both_data.append(np.concatenate([red_data[i], blue_data[i]]))

# %%
txt = plot_boxplot(both_data, ["\n".join(labels[variant].split(" ")) for variant in Folders.keys()],  title="", colors=colors,
             swarmsize=3, ylabel="ELO score", figsize=(10,4), rotation=0, use_stick=False, fontsize=18);
utils.savefig("ELO", figure_folder, timestamp=False)

# %%
print(txt)

# %% [markdown]
# # Phase Transitions

# %%
for variant, folder in tqdm(folders):
    path = os.path.join(root_folder, variant, folder)
    if not os.path.exists(path + "/atomics_usage_per_bt.pk.pk"):
        tournament = utils.load_pickle(path + "/generational_tournament.pk")
        red_indices = utils.load_pickle(path + "/red_indices.pk")
        blue_indices = utils.load_pickle(path + "/blue_indices.pk")
        Atomics = []
        for gen_id in range(10):
            actions = {}
            for i in range(*red_indices[gen_id]):
                bt_actions = set()
                for j in range(*blue_indices[gen_id]):
                    actions_id = tournament[i,j]["eval"]["actions_id"]
                    for a_id, n in zip(*actions_id["red"]):
                        bt_actions.add(a_id)
                for a_id in bt_actions:
                    if a_id not in actions:
                        actions[a_id] = 0
                    actions[a_id] += 1 
            for j in range(*blue_indices[gen_id]):
                bt_actions = set()
                for i in range(*red_indices[gen_id]):
                    actions_id = tournament[i,j]["eval"]["actions_id"]
                    for a_id, n in zip(*actions_id["blue"]):
                        bt_actions.add(a_id)
                for a_id in bt_actions:
                    if a_id not in actions:
                        actions[a_id] = 0
                    actions[a_id] += 1 
            Atomics.append(actions)
        utils.save_pickle(path + "/atomics_usage_per_bt.pk", Atomics)

# %% [markdown]
# ## one all_used_actions for all

# %%
all_variants = ["full", "no_bootstrap", "no_bias", "handcrafted_behavior", "diversity_only", "quality_only"]

# %%
All_used_actions = []
variants = ["full", "no_bias", "handcrafted_behavior"]
for i, (variant, folder) in enumerate(folders):
    if variant in variants:
        path = os.path.join(root_folder, variant, folder)
        Atomics = utils.load_pickle(path + "/atomics_usage_per_bt.pk")
        all_used_actions = list(set(np.concatenate([list(actions.keys()) for actions in Atomics])))
        All_used_actions += all_used_actions

all_used_actions = list(set(All_used_actions))
all_used_actions.remove(150)  # stand

global_usage = np.zeros((10, len(all_used_actions)))
global_elites = np.zeros((10, len(all_used_actions)))
for (variant, folder) in folders:
    if variant in variants:
        path = os.path.join(root_folder, variant, folder)
        Atomics = utils.load_pickle(path + "/atomics_usage_per_bt.pk")
        red_indices = utils.load_pickle(path + "/red_indices.pk")
        blue_indices = utils.load_pickle(path + "/blue_indices.pk")
        for i in range(10):
            for (a_id, n) in Atomics[i].items():
                if a_id != 150:  # stand
                    j = all_used_actions.index(a_id)
                    ri, rj = red_indices[i]
                    bi, bj = blue_indices[i]
                    global_usage[i,j] += n
                    global_elites[i,j] += (rj-ri + bj-bi)

global_usage = 100 * global_usage/global_elites
global_usage_norm = np.copy(global_usage)
for j in range(len(all_used_actions)):
    if np.sum(global_usage[:, j]) > 0:
        global_usage_norm[:, j] = (global_usage[:, j] / np.sum(global_usage[:, j]))

global_indices = np.argsort(np.sum(global_usage_norm * np.arange(10)[:, np.newaxis], axis=0))

plt.subplots(figsize=(16, 6))
ticks_fontsize = 12
plt.axis("off")

for kind_idx, kind in enumerate(["attack", "move toward",  "move away_from", "set_target A", "go_to A", "heal"]):
    filter_indices = []
    for i, idx in enumerate(global_indices):
        if grammar.all_variants[all_used_actions[idx]].startswith(kind) and np.max(global_usage[:, idx]) > 5:
            filter_indices.append(idx)

    ax = plt.subplot2grid((2, 3), (kind_idx%2, kind_idx//2))
    scatter = plt.pcolor(global_usage[:, filter_indices].T)
    cbar = plt.colorbar()

    cbar.ax.tick_params(labelsize=ticks_fontsize)
    x = [j+0.5 for j in range(0, 10)]
    x_ticks = [f"{i}\n{i+1}" for i in range(0,20,2)]
    plt.xticks(x, x_ticks, fontsize=ticks_fontsize-4);
    y = [i+0.5 for i in range(len(filter_indices))]
    y_ticks = [f"{i} - {grammar.all_variants[all_used_actions[i]][len(kind):]}" for i in filter_indices]
    plt.yticks(y, y_ticks, fontsize=ticks_fontsize)
#    plt.xlabel("Generations")
    plt.title(kind, fontsize=16)
    plt.tight_layout()

# %% [markdown]
# ## Zoom on one

# %%
ticks_fontsize = 14

variant = "no_bias"
rep = 1
path = os.path.join(root_folder, variant, Folders[variant][rep])
Atomics = utils.load_pickle(path + "/atomics_usage_per_bt.pk")

all_used_actions = list(set(np.concatenate([list(actions.keys()) for actions in Atomics])))
all_used_actions.remove(150)  # stand

usage = np.zeros((10, len(all_used_actions)))
elites = np.zeros((10, len(all_used_actions)))
red_indices = utils.load_pickle(path + "/red_indices.pk")
blue_indices = utils.load_pickle(path + "/blue_indices.pk")
for i in range(10):
    for (a_id, n) in Atomics[i].items():
        if a_id != 150:  # stand
            j = all_used_actions.index(a_id)
            ri, rj = red_indices[i]
            bi, bj = blue_indices[i]
            usage[i,j] += n
            elites[i,j] += (rj-ri + bj-bi)
usage = 100 * usage/elites
usage_norm = np.copy(usage)

for j in range(len(all_used_actions)):
    if np.sum(usage[:, j]) > 0:
        usage_norm[:, j] = (usage[:, j] / np.sum(usage[:, j]))
indices = np.argsort(np.sum(usage_norm * np.arange(10)[:, np.newaxis], axis=0))

plt.subplots(figsize=(16, 6))
ticks_fontsize = 12
plt.axis("off")

for kind_idx, kind in enumerate(["attack", "move toward",  "move away_from", "set_target A", "go_to A", "heal"]):
    filter_indices = []
    for i, idx in enumerate(indices):
        if grammar.all_variants[all_used_actions[idx]].startswith(kind) and np.max(usage[:, idx]) > 0:
            filter_indices.append(idx)

    ax = plt.subplot2grid((2, 3), (kind_idx%2, kind_idx//2))
    scatter = plt.pcolor(usage[:, filter_indices].T)
    cbar = plt.colorbar()

    cbar.ax.tick_params(labelsize=ticks_fontsize)
    x = [j+0.5 for j in range(0, 10)]
    x_ticks = [f"{i}\n{i+1}" for i in range(0,20,2)]
    plt.xticks(x, x_ticks, fontsize=ticks_fontsize-4);
    y = [i+0.5 for i in range(len(filter_indices))]
    y_ticks = [f"{i} - {grammar.all_variants[all_used_actions[i]][len(kind):]}" for i in filter_indices]
    plt.yticks(y, y_ticks, fontsize=ticks_fontsize)
#    plt.xlabel("Generations")
    plt.title(kind, fontsize=16)
    plt.tight_layout()

# %%
fig, ax = plt.subplots(figsize=(16,9))
plt.axis("off")
ax = plt.subplot2grid((2,1), (0,0))
x = [i for i in range(10)]
interesting_atomics = [148, 147, 164]
for i, a_id in enumerate(interesting_atomics):
    color = colors[i]
    plt.plot(x, usage[:, a_id], lw=3, ls=line_styles[i], c=color, label=grammar.all_variants[all_used_actions[a_id]].replace("attack ", ""))

x_ticks = [f"{i}-{i+1}" for i in range(0,20,2)]
plt.xticks(x, x_ticks);
plt.ylabel("Elites usage (%)", fontsize=fontsize)
#plt.xlabel("Generations")
plt.legend(title="(a) Attack", frameon=False, bbox_to_anchor=(0.95,1.2,0,0), title_fontproperties={'weight': 'bold', 'size': fontsize}, fontsize=fontsize)
ax.spines['left'].set_position(('data', 0))
plt.tight_layout()

# %% [markdown]
# ## variance along replications

# %%
All_used_actions = []
variants = ["full"]
for i, (variant, folder) in enumerate(folders):
    if variant in variants:
        path = os.path.join(root_folder, variant, folder)
        Atomics = utils.load_pickle(path + "/atomics_usage_per_bt.pk")
        all_used_actions = list(set(np.concatenate([list(actions.keys()) for actions in Atomics])))
        All_used_actions += all_used_actions
all_used_actions = list(set(All_used_actions))
all_used_actions.remove(150)  # stand

usage = np.zeros((10, len(all_used_actions), len(variants)*3))
k = -1
for (variant, folder) in folders:
    if variant in variants:
        k += 1 
        path = os.path.join(root_folder, variant, folder)
        Atomics = utils.load_pickle(path + "/atomics_usage_per_bt.pk")
        red_indices = utils.load_pickle(path + "/red_indices.pk")
        blue_indices = utils.load_pickle(path + "/blue_indices.pk")
        for i in range(10):
            for (a_id, n) in Atomics[i].items():
                if a_id != 150:  # stand
                    j = all_used_actions.index(a_id)
                    ri, rj = red_indices[i]
                    bi, bj = blue_indices[i]
                    usage[i,j,k] = 100*n/(rj-ri + bj-bi)

# %% [raw]
# global_usage = np.zeros((10, len(all_used_actions)))
# for i in range(10):
#     for j in range(len(all_used_actions)):
#         global_usage[i,j] = np.median(usage[i,j])
# plt.subplots(figsize=(16, 6))
# ticks_fontsize = 12
# plt.axis("off")
#
# for kind_idx, kind in enumerate(["attack", "move toward",  "move away_from", "set_target A", "go_to A", "heal"]):
#     filter_indices = []
#     for i, idx in enumerate(global_indices):
#         if grammar.all_variants[all_used_actions[idx]].startswith(kind) and np.max(global_usage[:, idx]) > 1:
#             filter_indices.append(idx)
#
#     ax = plt.subplot2grid((2, 3), (kind_idx%2, kind_idx//2))
#     scatter = plt.pcolor(global_usage[:, filter_indices].T)
#     cbar = plt.colorbar()
#
#     cbar.ax.tick_params(labelsize=ticks_fontsize)
#     x = [j+0.5 for j in range(0, 10)]
#     x_ticks = [f"{i}\n{i+1}" for i in range(0,20,2)]
#     plt.xticks(x, x_ticks, fontsize=ticks_fontsize-4);
#     y = [i+0.5 for i in range(len(filter_indices))]
#     y_ticks = [f"{i} - {grammar.all_variants[all_used_actions[i]][len(kind):]}" for i in filter_indices]
#     plt.yticks(y, y_ticks, fontsize=ticks_fontsize)
# #    plt.xlabel("Generations")
#     plt.title(kind, fontsize=16)
#     plt.tight_layout()

# %% [markdown]
# ### Global trends

# %%
fontsize = 30

fig, ax = plt.subplots(figsize=(16,9))
plt.axis("off")
ax = plt.subplot2grid((2,1), (0,0))
x = [i for i in range(10)]
interesting_atomics = [11, 29, 17, 23, 5]
for i, a_id in enumerate(interesting_atomics):
    color = colors[i]
    plt.plot(x, np.median(usage[:, a_id], axis=1), lw=3, ls=line_styles[i], c=color, label=grammar.all_variants[all_used_actions[a_id]].replace("attack ", ""))
    plt.fill_between(x, np.quantile(usage[:, a_id],0.25,  axis=1), np.quantile( usage[:, a_id], 0.75,axis=1), lw=3, color=color, alpha=0.3)

x_ticks = [f"{i}-{i+1}" for i in range(0,20,2)]
plt.xticks(x, x_ticks);
plt.ylabel("Elites usage (%)", fontsize=fontsize)
#plt.xlabel("Generations")
plt.legend(title="(a) Attack", frameon=False, bbox_to_anchor=(0.95,1.2,0,0), title_fontproperties={'weight': 'bold', 'size': fontsize}, fontsize=fontsize)
ax.spines['left'].set_position(('data', 0))
plt.tight_layout()

ax = plt.subplot2grid((2,1), (1,0))

x = [i for i in range(10)]
interesting_atomics = [180, 181, 182, 183]
for i, a_id in enumerate(interesting_atomics):
    color = colors[i]
    plt.plot(x, np.median(usage[:, a_id], axis=1), lw=3, c=color,ls=line_styles[i], label=grammar.all_variants[all_used_actions[a_id]].replace("go_to A ", ""))
    plt.fill_between(x, np.quantile(usage[:, a_id],0.,  axis=1), np.quantile( usage[:, a_id], 1,axis=1), lw=3, color=color, alpha=0.3)
    #plt.fill_between(x, np.quantile(usage[:, a_id],0.05,  axis=1), np.quantile( usage[:, a_id], 0.95,axis=1), lw=3, color=color, alpha=0.1)

x_ticks = [f"{i}-{i+1}" for i in range(0,20,2)]
plt.xticks(x, x_ticks);
plt.ylabel("Elites usage (%)", fontsize=fontsize)
plt.xlabel("Generations", fontsize=fontsize)
ax.spines['left'].set_position(('data', 0))
plt.legend(title="(b) Go_to",frameon=False, bbox_to_anchor=(1.03,1.05,0, 0), title_fontproperties={'weight': 'bold', 'size': fontsize}, fontsize=fontsize)
plt.tight_layout()
utils.savefig("global_trends", figure_folder, timestamp=False)

# %% [markdown]
# ## Red vs Blues 

# %%
for variant, folder in tqdm(folders):
    if variant in ["full", "no_bias"]:
        path = os.path.join(root_folder, variant, folder)
        if not os.path.exists(path + "/blue_atomics_usage_per_bt.pk.pk"):
            tournament = utils.load_pickle(path + "/generational_tournament.pk")
            red_indices = utils.load_pickle(path + "/red_indices.pk")
            blue_indices = utils.load_pickle(path + "/blue_indices.pk")
            red_Atomics, blue_Atomics = [], []
            for gen_id in range(10):
                actions = {}
                for i in range(*red_indices[gen_id]):
                    bt_actions = set()
                    for j in range(*blue_indices[gen_id]):
                        actions_id = tournament[i,j]["eval"]["actions_id"]
                        for a_id, n in zip(*actions_id["red"]):
                            bt_actions.add(a_id)
                    for a_id in bt_actions:
                        if a_id not in actions:
                            actions[a_id] = 0
                        actions[a_id] += 1 
                red_Atomics.append(actions)
                actions = {}
                for j in range(*blue_indices[gen_id]):
                    bt_actions = set()
                    for i in range(*red_indices[gen_id]):
                        actions_id = tournament[i,j]["eval"]["actions_id"]
                        for a_id, n in zip(*actions_id["blue"]):
                            bt_actions.add(a_id)
                    for a_id in bt_actions:
                        if a_id not in actions:
                            actions[a_id] = 0
                        actions[a_id] += 1 
                blue_Atomics.append(actions)
            utils.save_pickle(path + "/blue_atomics_usage_per_bt.pk", blue_Atomics)
            utils.save_pickle(path + "/red_atomics_usage_per_bt.pk", red_Atomics)

# %%
ticks_fontsize = 14

variant = "no_bias"
rep = 0


usages = {}
for rep in [0,1,2]:
    path = os.path.join(root_folder, variant, Folders[variant][rep])
    red_Atomics = utils.load_pickle(path + "/red_atomics_usage_per_bt.pk")
    blue_Atomics = utils.load_pickle(path + "/blue_atomics_usage_per_bt.pk")
    for k, Atomics in enumerate([blue_Atomics, red_Atomics]):
        all_used_actions = list(set(np.concatenate([list(actions.keys()) for actions in Atomics])))
        all_used_actions.remove(150)  # stand
        
        usage = np.zeros((10, len(all_used_actions)))
        elites = np.zeros((10, len(all_used_actions)))
        red_indices = utils.load_pickle(path + "/red_indices.pk")
        blue_indices = utils.load_pickle(path + "/blue_indices.pk")
        for i in range(10):
            for j in range(len(all_used_actions)):
                elites[i,j] += (bj-bi) if k == 0 else (rj-ri)
            for (a_id, n) in Atomics[i].items():
                if a_id != 150:  # stand
                    j = all_used_actions.index(a_id)
                    ri, rj = red_indices[i]
                    bi, bj = blue_indices[i]
                    usage[i,j] += n  
        usage = 100 * usage/elites
        usages[(rep, k)] = usage
        usage_norm = np.copy(usage)
        for j in range(len(all_used_actions)):
            if np.sum(usage[:, j]) > 0:
                usage_norm[:, j] = (usage[:, j] / np.sum(usage[:, j]))
        indices = np.argsort(np.sum(usage_norm * np.arange(10)[:, np.newaxis], axis=0))
        
        plt.subplots(figsize=(16, 6))
        ticks_fontsize = 12
        plt.axis("off")
        
        for kind_idx, kind in enumerate(["attack", "move toward",  "move away_from", "set_target A", "go_to A", "heal"]):
            filter_indices = []
            for i, idx in enumerate(indices):
                if grammar.all_variants[all_used_actions[idx]].startswith(kind) and np.max(usage[:, idx]) > 10:
                    filter_indices.append(idx)
        
            ax = plt.subplot2grid((2, 3), (kind_idx%2, kind_idx//2))
            scatter = plt.pcolor(usage[:, filter_indices].T)
            cbar = plt.colorbar()
        
            cbar.ax.tick_params(labelsize=ticks_fontsize)
            x = [j+0.5 for j in range(0, 10)]
            x_ticks = [f"{i}\n{i+1}" for i in range(0,20,2)]
            plt.xticks(x, x_ticks, fontsize=ticks_fontsize-4);
            y = [i+0.5 for i in range(len(filter_indices))]
            y_ticks = [f"{i} - {all_used_actions[i]} - {grammar.all_variants[all_used_actions[i]][len(kind):]}" for i in filter_indices]
            plt.yticks(y, y_ticks, fontsize=ticks_fontsize)
        #    plt.xlabel("Generations")
            plt.title(f"{rep} {k} " + kind, fontsize=16)
            plt.tight_layout()

# %%
fontsize= 20
fig, ax = plt.subplots(figsize=(16,5))
x = [i for i in range(10)]
interesting_atomics = [11, 29, 17, 23, 5]

for rep in [0,1,2]:
    path = os.path.join(root_folder, variant, Folders[variant][rep])
    red_Atomics = utils.load_pickle(path + "/red_atomics_usage_per_bt.pk")
    blue_Atomics = utils.load_pickle(path + "/blue_atomics_usage_per_bt.pk")
    i = 0
    color = blue_color
    all_used_actions = list(set(np.concatenate([list(actions.keys()) for actions in blue_Atomics])))
    all_used_actions.remove(150)  # stand
    y = usages[rep, 0][:, all_used_actions.index(176)] if 176 in all_used_actions else [0]*10
    plt.plot(x, y,  lw=3, ls=line_styles[rep], c=color, label=f"Rep {rep} - Blue - " + grammar.all_variants[176])
    i = 1
    color = red_color
    all_used_actions = list(set(np.concatenate([list(actions.keys()) for actions in red_Atomics])))
    all_used_actions.remove(150)  # stand
    y = usages[rep, 1][:, all_used_actions.index(13)] if 13 in all_used_actions else [0]*10
    plt.plot(x, y, lw=3, ls=line_styles[rep], c=color, label=f"Rep {rep} - Red - " + grammar.all_variants[13])

x_ticks = [f"{i}-{i+1}" for i in range(0,20,2)]
plt.xticks(x, x_ticks);
plt.ylabel("Elites usage (%)", fontsize=fontsize)
#plt.xlabel("Generations")
plt.legend(title="Atomics", frameon=False, bbox_to_anchor=(0.95,0.9,0,0), title_fontproperties={'weight': 'bold', 'size': fontsize}, fontsize=fontsize)
ax.spines['left'].set_position(('data', 0))
plt.tight_layout()
utils.savefig("arm_race_archers", figure_folder, timestamp=False)

# %% [markdown]
# # Pretty plot for diversity

# %% [markdown]
# ## PCA for teaser figure

# %%
indices = utils.load_pickle(saved_computation_path + "indices.pk")
rev_indices = []
for name in folders:
    rev_indices.append([i for i in range(len(indices)) if indices[i][0]==name])

projection = utils.load_pickle(saved_computation_path + "projection.pk")
F_pca = utils.load_pickle(saved_computation_path + "F_pca.pk")

# %%
rep_id = 6
variant, folder = folders[rep_id]

points = projection[rev_indices[rep_id]]

# %%
n_bins = 5
bins = [[[] for _ in range(n_bins)] for _ in range(n_bins)]

x_m, x_M = np.min(points[:, 0]), np.max(points[:, 0])
y_m, y_M = np.min(points[:, 1]), np.max(points[:, 1])

for i, p in enumerate(points):
    if F_pca[i] == -100:
        x = int((n_bins-1) * (p[0]-x_m)/(x_M-x_m))
        y = int((n_bins-1) * (p[1]-y_m)/(y_M-y_m))
        
        bins[x][y].append(i) 

# %%
samples = []
n_samples_per_bin = 3
for i in range(n_bins):
    for j in range(n_bins):
        if len(bins[i][j]) > 0:
            samples += list(np.random.choice(bins[i][j], min(n_samples_per_bin,len(bins[i][j])), replace=False))
utils.save_pickle(video_folder+"samples2.pk", samples)

# %%
samples = utils.load_pickle(video_folder+"samples2.pk")  # samples.pk (= random in bin), samples2.pk (only blue 100%) 

# %%
plt.subplots(figsize=(10, 10))
plt.axis("off")

plt.xticks([]);
plt.yticks([]);
scatter = plt.scatter(points[:, 0], points[:, 1], s=2, c=F_pca[rev_indices[rep_id]], vmin=-100, vmax=100, alpha=0.2, cmap="coolwarm",)
plt.scatter(points[samples, 0], points[samples, 1], s=200, facecolor="None", edgecolor="black", marker="o", alpha=1)
for i in samples:
    if True:
        plt.text(points[i, 0]+0.01, points[i, 1]+0.01, str(i), fontsize=6, backgroundcolor="white")
#norm = scatter.norm  # Get the normalization from the scatter
#cbar = ColorbarBase(ax, cmap=scatter.cmap, norm=norm, orientation='horizontal')
#cbar.set_label('Fitness', fontsize=12)

plt.tight_layout()

# %% [markdown]
# ## Video

# %%
path = os.path.join(root_folder, variant, folder)
tournament = utils.load_pickle(path + "/generational_tournament.pk")
duels = []
blue_gen, red_gen = 0, 0
for gen_id in range(10):
    Elites = utils.load_pickle(path + f"/gen_{gen_id*2}/elites_{gen_id*2}.pk")
    n_reds = len(Elites["reds"])
    n_blues = len(Elites["blues"])
    for i in range(n_reds):
        for j in range(n_blues):
            duels.append(tournament[(red_gen+i, blue_gen+j)])
    blue_gen += n_blues
    red_gen += n_reds

# %%
config = get_config(parabellum_config, 0, None, n_cells=None, n_groups=n_groups, use_embedding=True)
Measures = []
for i in tqdm(samples):
    if not os.path.exists(video_folder + f"/{i}_0.mp4"):
        duel = duels[i]
        if duel["task"]["config"]["generation"] == "red":
            blues, reds = duel["task"]["config"], duel["candidate"]["value"]
        else:
            blues, reds = duel["candidate"]["value"], duel["task"]["config"]
        a_bt, e_bt, sectors, unit_types = get_jax_params(reds, blues, n_groups, group_size, config["evaluation_config"]["fixed_starting_sector"], config["evaluation_config"]["fixed_unit_types"])
        unit_starting_sectors = jnp.array([sectors])
        unit_types = jnp.array([unit_types], dtype=jnp.uint8)
        Measures.append(video_eval_bi_bts_fn([a_bt], [e_bt], unit_starting_sectors, unit_types, True, video_folder + f"/{i}"))
utils.save_pickle(video_folder+"/Measures2.pk", Measures)

# %%
Measures = utils.load_pickle(video_folder + "Measures2.pk")

# %%
Frames = {}
for idx in samples:
    Frames[idx] = extract_frames(video_folder + f"/{idx}_0.mp4")

# %%
image_size = 300
pad = 2
n_frames = 6
W = (image_size+2*pad) * n_frames
H = (image_size+2*pad)

font_size = 44  # Adjust this value to change text size
font = ImageFont.load_default().font_variant(size=font_size)

# %%
video_id = 99264

# %%
# gif 
Frames[video_id][0].save(video_folder+f'{video_id}.gif', save_all=True, append_images=Frames[video_id][1:], optimize=True, duration=50, loop=0)

# snapshots
image = Image.new("RGBA", (W, H))
draw = ImageDraw.Draw(image)
image.paste(Image.new("RGBA", image.size, "WHITE"), (0,0))

measures = Measures[samples.index(video_id)]
print(video_id, 100*(1-measures.ally_health[0]), 100*(1-measures.enemy_health[0]))
for j, t in enumerate(np.linspace(0, min(int(100 * measures.duration[0])+1,99), n_frames, dtype=int)):
    (x_pos, y_pos) = j, 0
    y = y_pos
    x = x_pos *(image_size+2*pad) 
    img = Frames[video_id][t]
    img = img.resize((image_size, image_size))
    image.paste(Image.new("RGBA", (image_size+2*pad, image_size+2*pad), "black"), (x, y))
    image.paste(img, (x+pad, y+pad))
    # Add text for the t value in the bottom right corner
    # Add text in a simpler way
    text = f"{t+1}"
    
    # Calculate position for bottom right of each image
    text_x = x + image_size - (28 if t<10 else (55 if t < 99 else 79))
    text_y = y + 0
    
    # Draw the text
    draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)  # Black text, no font specified
image.save(video_folder+f"{video_id}.png")
image

# %%
displayed = {
    "a": 11223,  # 1
    "b": 28761,  # 1
    "c": 86200,  # 1
    "d": 99264,  # 2
    "e": 37488,  # 1
    "f": 20348,  # 1
    "g": 4438,  # 1
    "h": 9655,  # 1
    "i": 57756,  # 1
    "j": 82443,  # 1
    "k": 98381,  # 1
    "l": 61834,  # 1
    "m": 14534,  # 1
    "n": 55493,  # 1
    "o": 77984,  # 2
}

# %%
for key, video_id in displayed.items():
    if video_id in samples:
        with open(video_folder + f"{key}.txt", "w") as f:
            measures = Measures[samples.index(video_id)]
            blue_ato = np.unique(measures.actions_id[:, 0, :160])
            red_ato = np.unique(measures.actions_id[:, 0, 160:])
            f.write(f"Video id: {key}\n")
            f.write(f"Tournament id: {video_id}\n")
            duel = duels[video_id]
            f.write(f"Red: fitness {100*(1-measures.ally_health[0])}\n")
            f.write(f"{duel['candidate']['value']['bt'].to_pretty_txt()}\n")
            f.write(f"used atomics: {[grammar.all_variants[i] for i in red_ato]}\n")
            f.write(f"\nBlue: fitness {100*(1-measures.enemy_health[0])}\n")
            f.write(f"{duel['task']['config']['bt'].to_pretty_txt()}\n")
            f.write(f"used atomics: {[grammar.all_variants[i] for i in blue_ato]}\n")
