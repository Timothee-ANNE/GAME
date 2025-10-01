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
import gc
import subprocess
import cv2
from PIL import Image, ImageDraw 
import umap.umap_ as umap  # pip install umap-learn
from scipy.spatial import cKDTree
from sklearn import preprocessing
import misc_plot
from matplotlib.figure import Figure
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from PIL import ImageFont
from kmodes.kmodes import KModes

# %%
from GAME import *

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


# %%
def get_deck_key(solution):
    l = list(solution["deck"])
    l.sort()
    return "_".join(l)


# %%
def compute_qd_score(behaviors, fitness, step_size=0.01, alpha=1., verbose=False):
    qd_score, coverage = np.empty(len(behaviors)), np.empty(len(behaviors))
    cells = {}
    for i in tqdm(range(len(behaviors))) if verbose else range(len(behaviors)): 
        key = tuple([int(x/step_size) if x != 1 else (int(1/step_size)-1) for x in behaviors[i]])  # make sure to have 1/step_size bins per dimension when behavior = max bound
        if key not in cells:
            cells[key] = fitness[i]
        else:
            cells[key] = max(cells[key], fitness[i])
        qd_score[i] = sum((x for x in cells.values())) * step_size**len(key)
        coverage[i] = len(cells)* step_size**len(key)
    return qd_score, coverage, cells

def normalize_B(B, all_B):
    return (B-np.min(all_B, axis=0))/(np.max(all_B, axis=0)-np.min(all_B, axis=0))

def get_bounds(all_B):
    return np.min(all_B, axis=0), np.max(all_B, axis=0)


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
game_xps = {
    "WARRIOR vs WARLOCK": {"path": "/home/tim/Experiments/data/2025/08/12/15h33m41s/", "red": CHARACTER_CLASS.WARRIOR, "blue": CHARACTER_CLASS.WARLOCK},
    "ROGUE vs PALADIN": {"path": "/home/tim/Experiments/data/2025/08/12/22h39m15s/", "red": CHARACTER_CLASS.ROGUE, "blue": CHARACTER_CLASS.PALADIN},
    "HUNTER vs DRUID": {"path": "/home/tim/Experiments/data/2025/08/13/04h58m22s/", "red": CHARACTER_CLASS.HUNTER, "blue": CHARACTER_CLASS.DRUID},
    "SHAMAN vs MAGE": {"path": "/home/tim/Experiments/data/2025/08/13/10h15m43s/", "red": CHARACTER_CLASS.SHAMAN, "blue": CHARACTER_CLASS.MAGE},
    "MAGE vs PRIEST": {"path": "/home/tim/Experiments/data/2025/08/13/15h30m05s/" , "red": CHARACTER_CLASS.MAGE, "blue": CHARACTER_CLASS.PRIEST},
}

# %%
me_xps = {
    "WARRIOR vs WARLOCK": {
        "blue_path": '/home/tim/Experiments/data/2025/08/15/15h06m10s/',  # Warlock vs Warrior 
        "red_path": '/home/tim/Experiments/data/2025/08/15/20h27m36s/',  # Warrior vs Warlock
        "red": CHARACTER_CLASS.WARRIOR, "blue": CHARACTER_CLASS.WARLOCK},
    "ROGUE vs PALADIN": {
        "blue_path": '/home/tim/Experiments/data/2025/08/15/22h26m46s/',  # Paladin vs Rogue 
        "red_path": '/home/tim/Experiments/data/2025/08/16/00h52m09s/',  # Rogue vs Paladin 
        "red": CHARACTER_CLASS.ROGUE, "blue": CHARACTER_CLASS.PALADIN},
    "HUNTER vs DRUID": {
        "blue_path": '/home/tim/Experiments/data/2025/08/16/05h10m41s/',  # Druid vs Hunter 
        "red_path": '/home/tim/Experiments/data/2025/08/16/07h45m25s/',  # Hunter vs Druid 
        "red": CHARACTER_CLASS.HUNTER, "blue": CHARACTER_CLASS.DRUID},
    "SHAMAN vs MAGE": {
        "blue_path": '/home/tim/Experiments/data/2025/08/16/10h54m35s/',  #  Mage vs Shaman
        "red_path": '/home/tim/Experiments/data/2025/08/16/13h58m55s/',  #  Shaman vs Mage
        "red": CHARACTER_CLASS.SHAMAN, "blue": CHARACTER_CLASS.MAGE},
    "MAGE vs PRIEST": {
        "blue_path": '/home/tim/Experiments/data/2025/08/16/16h40m24s/',  # Priest vs Mage
        "red_path": '/home/tim/Experiments/data/2025/08/16/19h24m48s/',  # Mage vs Priest
        "red": CHARACTER_CLASS.MAGE, "blue": CHARACTER_CLASS.PRIEST},
}

# %% [markdown]
# ## Deckbuilding

# %% [markdown]
# ### Card distributions

# %%
Elites = {}
for xp_name, xp in game_xps.items():
    print(xp_name)
    path = xp["path"]
    game_xp = game_xps[xp_name]
    me_xp = me_xps[xp_name]

    red_game_archives = utils.load_pickle(path + f"gen_{6}/archive_save.pk")
    blue_game_archives = utils.load_pickle(path + f"gen_{7}/archive_save.pk")

    elites = {"red": [], "blue": []}
    archives = red_game_archives
    for archives, name in zip([red_game_archives, blue_game_archives], ["red", "blue"]):
        for archive in archives["solutions"]:
            for sol in archive.values():
                elites[name].append(list(sol["deck"]))
    Elites[xp_name] = elites


# %%
card = valid_cards['Stormwind Knight']()

# %%
np.mean([card.is_minion() for card in valid_cards.values()])

# %%
count = [np.mean([valid_cards[card_name]().is_minion() for card_name in deck]) for deck in Elites[xp_name][side] for xp_name in game_xps.keys() for side in ["red", "blue"] ]

# %%
print(np.min(count), np.quantile(count, 0.05), np.quantile(count, 0.25), np.median(count), np.quantile(count, 0.75), np.quantile(count, 0.95), np.max(count))

# %% [markdown]
# ### Load

# %% [markdown]
# #### ME 

# %%
for xp in me_xps.values():
    for side in ["red", "blue"]:
        path = xp[f"{side}_path"]
        archive = utils.load_pickle(path+"archive_save_40000.pk")  # 40k for one 40k for the other = 80k
        dic = {}
        dic["archive"] = archive
        dic["F"] = archive["fitness"][0]
        dic["B"] = archive["behavior"][0]
        xp[f"{side}_me"] = dic 

# %% [markdown]
# #### Load GAME archives

# %%
n_gen = 8

for xp in game_xps.values():
    game_folder = xp["path"]
    F_pca, B_pca, F, indices = [], [], {}, []
    Elites = [utils.load_pickle(game_folder + f"/gen_{gen_id}/elites_{gen_id}.pk") for gen_id in range(n_gen)]
    
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
    red_B = {}
    blue_B = {}
    F = {}
    for blue_gen_id, (blue_i, blue_j) in blue_indices.items():
        for red_gen_id, (red_i, red_j) in red_indices.items():
            mini_tournament = utils.load_pickle(game_folder + f"generational_tournament_{blue_gen_id}_{red_gen_id}.pk")
            for (i, j), duel in mini_tournament.items():
                F_red[i,j] = (duel["eval"]["fitness"]+30)/60
                red_B[i,j] = duel["eval"]["behavior"]
                blue_B[i,j] = duel["eval"]["other_behavior"]
    xp["F"] = F_red
    xp["red_B"] = red_B
    xp["blue_B"] = blue_B

# %% [markdown]
# ### Elo through generations

# %%
plt.subplots(figsize=(14,10))
plt.axis("off")
for xp_id, (name, xp) in enumerate(game_xps.items()):
    red_ratings, blue_ratings = asymmetric_elo(xp["F"])
    n_tasks = 50
    sorted_red_indices = np.concatenate([n_tasks*i+np.argsort(np.mean(xp["F"][i*n_tasks:(i+1)*n_tasks], axis=1)) for i in range(n_gen//2)])
    sorted_blue_indices = np.concatenate([n_tasks*i+np.argsort(-np.mean(xp["F"][:, i*n_tasks:(i+1)*n_tasks], axis=0))  for i in range(n_gen//2)])
    
    plt.subplot2grid((2,3), (xp_id%2, xp_id//2))
    plt.pcolor(xp["F"][sorted_red_indices][:, sorted_blue_indices], vmin=0, vmax=1, cmap=cm.coolwarm)
    plt.xticks([(i+j)/2 for (i,j) in blue_indices.values()], blue_indices.keys())
    plt.yticks([(i+j)/2 for (i,j) in red_indices.values()], red_indices.keys())
    plt.xlabel(classes_names[xp["blue"]])
    plt.ylabel(classes_names[xp["red"]])


# %%
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA 
import random 
from scipy.stats import ttest_ind, ttest_rel, shapiro, mannwhitneyu, pearsonr, wilcoxon
from colorama import Fore
from IPython.display import HTML
import seaborn as sb
import pandas as pd
from matplotlib import animation, rc
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm
from ipywidgets import interact
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import matplotlib.patheffects as PathEffects


# %%
def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def compute_test(data, coef=1, N=5):
    pairs = []
    for i in range(N-1):
        for j in range(i+1,N):
            pairs.append((i,j))
    res = []
    cells = [[ '' for _ in range(N-1)] for _ in range(N-1)]
    for (param1,param2) in pairs:
        data1 = np.array(data[param1])
        data2 = np.array(data[param2])
        stat, p_t = mannwhitneyu(data1,data2)  # mannwhitneyu does not make the normal hypothesis 
        p_t = p_t*coef
        mean1 = np.median(data1)
        mean2 = np.median(data2)
        if mean2 != 0:
            ratio = mean1/mean2
            d = truncate(ratio, 2)
            if d=='0.00' or d=='-0.00':
                d = truncate(ratio, 3)
                if d=='0.000' or d=='-0.000':
                    d = truncate(ratio, 4)
        else:
            d = "∞"
                
        if p_t < 0.001:
            res.append((param1, param2, '***'))
            cells[param1][N-1-param2] = d + '\n***'
        elif p_t < 0.01:
            res.append((param1, param2, '**'))
            cells[param1][N-1-param2] = d + '\n**'
        elif p_t < 0.05:
            res.append((param1, param2, '*'))
            cells[param1][N-1-param2] = d + '\n*'
        else:
            res.append((param1, param2, "ns"))
            cells[param1][N-1-param2] = "ns"
    return res, cells


# %%
def plot_boxplot(data, names, ylabel="performance", ylim=None, title="", log=False, bbox=(1.13,0.1,0.5,0.9), cmap=cm.rainbow, figsize=(16,9), fig=None, ax=None,
                 correction=True, rotation=0, use_table=False, use_stick=False, swarmsize=7, force_swarm=False, swarmdata=None, colors=None, verticale=True, x_ticks=True, y_ticks=True, xlabel=None, fontsize=18, titlefontsize=20):
    N = len(data)
    stat, cells = compute_test(data, coef=3*N*(N-1)/2 if correction else 1, N=N)
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    bp = ax.boxplot(x=data, positions=range(N), vert=verticale)
    colors = [int_to_color(i, N, cmap=cmap) for i in range(N)] if colors is None else colors 
    
    if np.sum([len(x) for x in data]) <= 100*N or force_swarm:
        sb.swarmplot(data=swarmdata if swarmdata is not None else data, color='black', edgecolor='black', size=swarmsize, dodge=True, orient="v" if verticale else "h")
    plt.grid(axis='y' if verticale else "x", alpha=0.3)
    if True:
        for i in range(0, len(bp['boxes'])):
            bp['boxes'][i].set_color(colors[i])
            # we have two whiskers!
            bp['whiskers'][i*2].set_color(colors[i])
            bp['whiskers'][i*2 + 1].set_color(colors[i])
            bp['whiskers'][i*2].set_linewidth(2)
            bp['whiskers'][i*2 + 1].set_linewidth(2)
            # fliers
            # (set allows us to set many parameters at once)
            bp['fliers'][i].set(markerfacecolor=colors[i],
                           marker='o', alpha=0.75, markersize=6,
                           markeredgecolor='none')
            bp['medians'][i].set_color('black')
            bp['medians'][i].set_linewidth(3)
            # and 4 caps to remove
            for c in bp['caps']:
                c.set_linewidth(0)

        for i in range(len(bp['boxes'])):
            box = bp['boxes'][i]
            box.set_linewidth(0)
            boxX = []
            boxY = []
            for j in range(5):
                boxX.append(box.get_xdata()[j])
                boxY.append(box.get_ydata()[j])
                boxCoords = list(zip(boxX,boxY))
                boxPolygon = Polygon(boxCoords, facecolor = colors[i], linewidth=0)
                ax.add_patch(boxPolygon)
        
    rows = names[:N-1]
    columns = [names[i] for i in range(N-1,0,-1)]   
    txt = ""
    if use_table:
        cell_text = cells
        cellColours = [['white' if N-1-i>j else 'lightgrey' for j in range(N-1)] for i in range(N-1) ]
        the_table = plt.table(cellText=cell_text,
                              rowLabels=rows,
                              cellColours= cellColours,
                              rowColours=colors[:N-1],
                              colColours=[ colors[i] for i in range(N-1,0,-1)],
                              colLabels=columns,
                              cellLoc = 'center',
                              bbox=bbox)
   
    
    txt += "| |"+"|".join([x.replace("\n", " ") for x in columns])+"|\n"
    txt += "|:-:|"+"|".join([':-:']*len(columns))+"|\n"
    for i in range(len(rows)):
        row = [ ( c.split("\n")[-1] if "\n" in c else c) for c in cells[i]]
        txt += "|" + rows[i].replace("\n", " ")+ "|" + "|".join(row) + '|\n'
    if use_stick and N == 2:
        maxi, mini = np.max([np.max(x) for x in data]), np.min([np.min(x) for x in data])
        top, bot, toptop = maxi + (maxi-mini)*0.05, maxi + (maxi-mini)*0.02, maxi + (maxi-mini)*0.06
        plt.plot([0,0,1,1], [bot, top, top, bot], color ="black")
        plt.text(s=stat[0][2], x=0.5, y=toptop, ha="center", fontsize=fontsize)
    if x_ticks:
        (plt.xticks if verticale else plt.yticks)(range(N), names, rotation=rotation, fontsize=fontsize)
    else:
        (plt.xticks if verticale else plt.yticks)([])
    if log:
        plt.yscale('log')
    if not ylim is None:
        plt.ylim(ylim)
    if not y_ticks:
        (plt.yticks if verticale else plt.xticks)([])
    else:
        (plt.yticks if verticale else plt.xticks)(fontsize=fontsize)
    if ylabel:
        plt.ylabel(ylabel, fontsize=fontsize) if verticale else plt.xlabel(ylabel, fontsize=fontsize)
    if xlabel:
        plt.xlabel(xlabel, fontsize=fontsize) if verticale else plt.ylabel(xlabel, fontsize=fontsize)
    plt.title(title, fontsize=titlefontsize, fontweight='bold')
    txt += "\n"
    for i in range(N):
        txt += f"{names[i]}: {np.median(data[i]):2.1f} [{np.quantile(data[i], 0.25)}, {np.quantile(data[i], 0.75)}]\n"
    txt += "\n"
    for i in range(N):
        txt += f"{names[i]}: {np.median(data[i]):2.1f} [{np.quantile(data[i], 0.25):2.1f}, {np.quantile(data[i], 0.75):2.1f}]\n"
    plt.tight_layout()
    return txt 

# %%
fig = plt.subplots(figsize=(20,5))
plt.axis("off")
for xp_id, (name, xp) in enumerate(game_xps.items()):
    red_ratings, blue_ratings = asymmetric_elo(xp["F"])
    
    ax = plt.subplot2grid((2, 5), (0, xp_id))
    n_tasks = 50
    
    data = [red_ratings[i*n_tasks:(i+1)*n_tasks] for i in range(n_gen//2)]
    best = np.argmax([np.median(x) for x in data])
    names = [str(i) for i in range(0,n_gen,2) ]
    plot_boxplot(data, names, title=f"({['a', 'b', 'c', 'd', 'e'][xp_id]})\n"+classes_names[xp["red"]], swarmsize=1, ylabel="ELO Score" if xp_id==0 else "",  xlabel="Generations", fig=fig, ax=ax, cmap="Reds", fontsize=18, titlefontsize=18);
    
    ax = plt.subplot2grid((2, 5), (1, xp_id))
    data = [blue_ratings[i*n_tasks:(i+1)*n_tasks] for i in range(n_gen//2)]
    best = np.argmax([np.median(x) for x in data])
    names = [str(i) for i in range(1,n_gen,2) ]
    plot_boxplot(data, names, title=classes_names[xp["blue"]], swarmsize=1, ylabel="ELO Score" if xp_id==0 else "", xlabel="Generations", fig=fig, ax=ax, cmap="Blues", fontsize=18, titlefontsize=18);
plt.tight_layout()
plt.savefig(figure_folder + "hearthstone_ELO_generations.pdf")

# %% [markdown]
# ## QD MAP

# %%
plt.subplots(figsize=(20,8))
plt.axis("off")

foo = np.mean

for xp_id, (name, xp) in enumerate(game_xps.items()):
    red_B, red_F = [], []
    blue_B, blue_F = [], []
    for i in range(n_tasks*n_gen//2):
        j = 0
        while j < n_tasks*n_gen//2 and (i,j) not in xp["B"]:
            j+=1
        if (i,j) in xp["red_B"]:
            red_B.append(xp["red_B"][i,j])
            red_F.append(foo(xp["F"][i]))
    red_B = np.array(red_B)
    for j in range(n_tasks*n_gen//2):
        i = 0
        while i < n_tasks*n_gen//2 and (i,j) not in xp["B"]:
            i += 1
        if (i,j) in xp["blue_B"]:
            blue_B.append(xp["blue_B"][i,j])
            blue_F.append(foo(xp["F"][:, j]))
    blue_B = np.array(blue_B)

    ax = plt.subplot2grid((2,5), (0, xp_id))
    plt.scatter(red_B[:,0], red_B[:,1], c=red_F, vmin=0, vmax=1, cmap="coolwarm") 
    plt.xticks([])
    plt.yticks([])

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.title(f"{classes_names[xp['red']]}", color=red_color)
    plt.tight_layout(pad=0)

    ax = plt.subplot2grid((2,5), (1, xp_id))
    plt.scatter(blue_B[:,0], blue_B[:,1], c=blue_F, vmin=0, vmax=1, cmap="coolwarm") 
    plt.xticks([])
    plt.yticks([])

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.title(f"vs {classes_names[xp['blue']]}", color=blue_color)
    plt.tight_layout(pad=0)

# %%

# %% [markdown]
# ### Coverage and QD

# %%
B_hunter_me = hunter_me["behavior"][0]
B_mage_me = mage_me["behavior"][0]

F_hunter_me = hunter_me["fitness"][0]
F_mage_me = mage_me["fitness"][0]

# %%
B_hunter_game = np.array(hunter_game["behavior"]).reshape((1000,2))
B_mage_game = np.array(mage_game["behavior"]).reshape((1000,2))

F_hunter_game = np.array(hunter_game["fitness"]).reshape((1000))
F_mage_game = np.array(mage_game["fitness"]).reshape((1000))

# %% [raw]
# all_B = np.concatenate([B_hunter_me, B_mage_me])

# %%
all_B = np.concatenate([B_hunter_me, B_mage_me, B_hunter_game, B_mage_game])

# %%
plt.subplots(figsize=(4*3, 4*2.5))
plt.axis("off")

xmin, ymin = np.min(all_B, axis=0)
xmax, ymax = np.max(all_B, axis=0)
xpad = (xmax-xmin)*0.05
ypad = (ymax-ymin)*0.05

cmap_name = "viridis"

plt.subplot2grid((2, 21), (0, 0), colspan=10)
scatter = plt.scatter(B_hunter_game[:,0], B_hunter_game[:,1], s=2, c=F_hunter_game, vmin=-30, vmax=30, alpha=1, cmap=cmap_name,)
plt.xlim((xmin-xpad, xmax+xpad))
plt.ylim((ymin-ypad, ymax+ypad))
plt.ylabel("GAME\nMana cost variation")
plt.title("Hunter")

plt.subplot2grid((2, 21), (0, 10), colspan=10)
scatter = plt.scatter(B_mage_game[:,0], B_mage_game[:,1], s=2, c=F_mage_game, vmin=-30, vmax=30, alpha=1, cmap=cmap_name,)
plt.xlim((xmin-xpad, xmax+xpad))
plt.ylim((ymin-ypad, ymax+ypad))
plt.title("Mage")

plt.subplot2grid((2, 21), (1, 0), colspan=10)
scatter = plt.scatter(B_hunter_me[:,0], B_hunter_me[:,1], s=2, c=F_hunter_me, vmin=-30, vmax=30, alpha=1, cmap=cmap_name,)
plt.xlim((xmin-xpad, xmax+xpad))
plt.ylim((ymin-ypad, ymax+ypad))
plt.ylabel("ME\nMana cost variation")
plt.xlabel("Mana cost mean")

plt.subplot2grid((2, 21), (1, 10), colspan=10)
scatter = plt.scatter(B_mage_me[:,0], B_mage_me[:,1], s=2, c=F_mage_me, vmin=-30, vmax=30, alpha=1, cmap=cmap_name,)
plt.xlim((xmin-xpad, xmax+xpad))
plt.ylim((ymin-ypad, ymax+ypad))
plt.xlabel("Mana cost mean")

ax_colorbar = plt.subplot2grid((2, 21), (0, 20), rowspan=2, colspan=1)
norm = scatter.norm  # Get the normalization from the scatter
cbar = ColorbarBase(ax_colorbar, cmap=scatter.cmap, norm=norm, orientation='vertical')
cbar.set_label('Fitness', fontsize=44)

plt.tight_layout()
#plt.savefig(f"{figure_folder}PCAs.png") 

# %% [markdown]
# ### ME through steps

# %% [raw]
# coverage = {"mage": [], "hunter": []}
# qd_score = {"mage": [], "hunter": []}
#
# for steps in range(1, 6):
#     steps *= 20000
#     hunter_me = utils.load_pickle(hunter_folder+f"archive_save_{steps}.pk")
#     mage_me = utils.load_pickle(mage_folder+f"archive_save_{steps}.pk")
#     B_hunter_me = hunter_me["behavior"][0]
#     B_mage_me = mage_me["behavior"][0]
#     
#     F_hunter_me = hunter_me["fitness"][0]
#     F_mage_me = mage_me["fitness"][0]
#
#     all_B = [[0,0], [10, 10]]
#     mage_me_qd, mage_me_cov, _ = compute_qd_score(normalize_B(B_mage_me, all_B), F_mage_me, 0.01)
#     hunter_me_qd, hunter_me_cov, _ = compute_qd_score(normalize_B(B_hunter_me, all_B), F_hunter_me, 0.01)
#     coverage["mage"].append(mage_me_cov[-1])
#     coverage["hunter"].append(hunter_me_cov[-1])
#     qd_score["mage"].append(mage_me_qd[-1])
#     qd_score["hunter"].append(hunter_me_qd[-1])


# %% [raw]
# plt.plot(qd_score["mage"], label="mage")
# plt.plot(qd_score["hunter"], label="hunter")
# plt.legend()

# %% [markdown]
# ### GAME against starting decks

# %% [raw]
# for xp_name, xp in game_xps.items():
#     path = xp["path"]
#     for (side, gen_id) in [("blue", 7), ("red", 6)]:
#         print(xp_name, side)
#         archives = utils.load_pickle(path + f"gen_{gen_id}/archive_save.pk")
#         reds = [deck for archive in archives["solutions"] for deck in archive.values()]
#         blue_hero = xp["red" if side == "blue" else "blue"]
#         blues = [{"deck": classes_starting_deck[blue_hero], "hero_class": blue_hero}]
#         config = get_config(hearthstone_config, 0, None, 0, "red", seed_id=0)
#         tournament = compute_tournament(config, reds, blues)
#         utils.save_pickle(path + f"{side}_game_against_starting_deck.pk", {"tournament": tournament, "reds": reds, "blues": blues})

# %%
for xp_name, xp in game_xps.items():
    path = xp["path"]
    for side in ["blue", "red"]:
        dic = utils.load_pickle(path + f"{side}_game_against_starting_deck.pk")
        F = []
        B = []
        for key, ev in dic["tournament"].items():
            F.append(ev["eval"]["fitness"])
            B.append(ev["eval"]["behavior"])
        dic["F"] = np.array(F)
        dic["B"] = np.array(B)
        xp[f"{side}_game_against_starting_deck"] = dic

# %%
step_size = 0.05

for xp_name in game_xps.keys():
    game_xp = game_xps[xp_name]
    me_xp = me_xps[xp_name]
    all_B = np.concatenate([me_xp["red_me"]["B"], me_xp["blue_me"]["B"], game_xp["red_game_against_starting_deck"]["B"], game_xp["blue_game_against_starting_deck"]["B"]])
    for side in ["red", "blue"]:
        for variant in ["GAME", "ME"]:
            if variant == "GAME":
                B = game_xp[f"{side}_game_against_starting_deck"]["B"]
                F = game_xp[f"{side}_game_against_starting_deck"]["F"]
            else:
                B = me_xp[f"{side}_me"]["B"]
                F = me_xp[f"{side}_me"]["F"]
            qd, cov, cells = compute_qd_score(normalize_B(B, all_B), (np.array(F)+30)/60, step_size)
            if variant == "GAME":
                game_xp[f"{side}_game_against_starting_deck"]["QD_score"] = qd[-1]
                game_xp[f"{side}_game_against_starting_deck"]["coverage"] = cov[-1]
                game_xp[f"{side}_game_against_starting_deck"]["cells"] = cells
                game_xp[f"{side}_game_against_starting_deck"]["best"] = np.max(F)
            else:
                me_xp[f"{side}_me"]["QD_score"] = qd[-1]
                me_xp[f"{side}_me"]["coverage"] = cov[-1]
                me_xp[f"{side}_me"]["cells"] = cells
                me_xp[f"{side}_me"]["best"] = np.max(F)

# %%
for what in ["coverage"]:
    print(what)
    for side in ["red", "blue"]:
        print("\t", side)
        Cov = [ f"{game_xps[xp_name][f'{side}_game_against_starting_deck'][what]*100:2.1f}%" for xp_name in game_xps.keys()]
        print("\t\t","GAME", Cov)
        Cov = [ f"{me_xps[xp_name][f'{side}_me'][what]*100:2.1f}%" for xp_name in game_xps.keys()]
        print("\t\t","ME", Cov)
    
for what in ["QD_score", "best"]:
    print(what)
    for side in ["red", "blue"]:
        print("\t", side)
        Cov = [ f"{game_xps[xp_name][f'{side}_game_against_starting_deck'][what]:2.1f}" for xp_name in game_xps.keys()]
        print("\t\t","GAME", Cov)
        Cov = [ f"{me_xps[xp_name][f'{side}_me'][what]:2.1f}" for xp_name in game_xps.keys()]
        print("\t\t","ME", Cov)

# %%
n = int(1/step_size)
n_sticks = 5

for xp_name in game_xps.keys():
    game_xp = game_xps[xp_name]
    me_xp = me_xps[xp_name]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Increased width
    #plt.axis("off")
    
    all_B = np.concatenate([me_xp["red_me"]["B"], me_xp["blue_me"]["B"], game_xp["red_game_against_starting_deck"]["B"], game_xp["blue_game_against_starting_deck"]["B"]])
    B_m, B_M = get_bounds(all_B)
    for i, side in enumerate(["red", "blue"]):
        for j, variant in enumerate(["GAME", "ME"]):
            #plt.subplot2grid((2, 21), (j, 10*i), colspan=10)
            ax = axes[j, i]  # Use the specific axis
            if variant == "GAME":
                cells = game_xp[f"{side}_game_against_starting_deck"]["cells"]
            else:
                cells = me_xp[f"{side}_me"]["cells"]
                
            
            M = np.zeros((n,n))
            for (x,y), f in cells.items():
                M[x,y] = f
            im = ax.pcolor(M.T, vmin=0, vmax=1)
            x = np.linspace(0, 1, n_sticks)

            ax.set_xticks(x*n)
            ax.set_xticklabels([f'{v:1.1f}' for v in x*(B_M[0]-B_m[0])+B_m[0]])
            ax.set_yticks(x*n)
            ax.set_yticklabels([f'{v:1.1f}' for v in x*(B_M[1]-B_m[1])+B_m[1]])
            
            if j == 0:
                ax.set_title(classes_names[game_xp[side]])
            else:
                ax.set_xlabel("Mana cost mean")
            if i == 0:
                ax.set_ylabel(f"{variant}\nMana cost variation")
    
    plt.tight_layout()  # Now this will work properly

    fig.subplots_adjust(right=0.85)  # Make space for colorbar
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Fitness', fontsize=12)


# %%
# with dots (scatter plots no real archive = grid)

for xp_name in game_xps.keys():

    game_xp = game_xps[xp_name]
    me_xp = me_xps[xp_name]
    
    plt.subplots(figsize=(4*3, 4*2.5))
    plt.axis("off")
    
    all_B = np.concatenate([me_xp["red_me"]["B"], me_xp["blue_me"]["B"], game_xp["red_game_against_starting_deck"]["B"], game_xp["blue_game_against_starting_deck"]["B"]])
    xmin, ymin = np.min(all_B, axis=0)
    xmax, ymax = np.max(all_B, axis=0)
    xpad = (xmax-xmin)*0.05
    ypad = (ymax-ymin)*0.05
    
    for i, side in enumerate(["red", "blue"]):
        for j, variant in enumerate(["GAME", "ME"]):
            plt.subplot2grid((2, 21), (j, 10*i), colspan=10)
            if variant == "GAME":
                B = game_xp[f"{side}_game_against_starting_deck"]["B"]
                F = game_xp[f"{side}_game_against_starting_deck"]["F"]
            else:
                B = me_xp[f"{side}_me"]["B"]
                F = me_xp[f"{side}_me"]["F"]
            scatter = plt.scatter(B[:,0], B[:,1], s=2, c=F, vmin=-30, vmax=30, alpha=1, cmap="coolwarm")
            plt.xlim((xmin-xpad, xmax+xpad))
            plt.ylim((ymin-ypad, ymax+ypad))
            if j == 0:
                plt.title(classes_names[game_xp[side]])
            else:
                plt.xlabel("Mana cost mean")
            if i == 0:
                plt.ylabel(f"{variant}\nMana cost variation")
    
    ax_colorbar = plt.subplot2grid((2, 21), (0, 20), rowspan=2, colspan=1)
    norm = scatter.norm  # Get the normalization from the scatter
    cbar = ColorbarBase(ax_colorbar, cmap=scatter.cmap, norm=norm, orientation='vertical')
    cbar.set_label('Fitness against starting deck', fontsize=24)

# %%
# Combined plot with all experiments as horizontal subplots
num_experiments = len(game_xps.keys())
fig = plt.figure(figsize=(4*2*num_experiments, 4*2))
plt.axis("off")

# Calculate global bounds across all experiments
all_B_global = []
for xp_name in game_xps.keys():
    game_xp = game_xps[xp_name]
    me_xp = me_xps[xp_name]
    all_B = np.concatenate([me_xp["red_me"]["B"], me_xp["blue_me"]["B"], 
                           game_xp["red_game_against_starting_deck"]["B"], 
                           game_xp["blue_game_against_starting_deck"]["B"]])
    all_B_global.append(all_B)

all_B_global = np.concatenate(all_B_global)
xmin, ymin = np.min(all_B_global, axis=0)
xmax, ymax = np.max(all_B_global, axis=0)
xpad = (xmax-xmin)*0.05
ypad = (ymax-ymin)*0.05

# Create subplots for each experiment
scatter = None  # Will store the last scatter plot for colorbar
for exp_idx, xp_name in enumerate(game_xps.keys()):
    game_xp = game_xps[xp_name]
    me_xp = me_xps[xp_name]
    
    # Each experiment gets 4 subplots (2x2) + space for colorbar (if last experiment)
    base_col = exp_idx * 20  # 21 cols per experiment + 1 spacing
    
    for i, side in enumerate(["red", "blue"]):
        for j, variant in enumerate(["GAME", "ME"]):
            plt.subplot2grid((2, num_experiments * 20), (j, base_col + 10*i), colspan=10)
            
            if variant == "GAME":
                B = game_xp[f"{side}_game_against_starting_deck"]["B"]
                F = game_xp[f"{side}_game_against_starting_deck"]["F"]
            else:
                B = me_xp[f"{side}_me"]["B"]
                F = me_xp[f"{side}_me"]["F"]
            
            scatter = plt.scatter(B[:,0], B[:,1], s=2, c=F, vmin=-30, vmax=30, alpha=1, cmap="viridis")
            plt.xlim((xmin-xpad, xmax+xpad))
            plt.ylim((ymin-ypad, ymax+ypad))
            
            if j == 0:
                a = "(" + "abcdefghijklm"[exp_idx*2+i] + ") "
                plt.title(a + classes_names[game_xp[side]], fontsize=24, fontweight="bold")
            else:
                plt.xlabel("Mana cost mean", fontsize=24)
            if i == 0 and exp_idx == 0:
                plt.ylabel(f"{variant}\nMana cost variation", fontsize=22)

# Add colorbar only once at the rightmost position
ax_colorbar = plt.subplot2grid((2, num_experiments * 20), (0, num_experiments * 20 - 1), rowspan=2, colspan=1)
norm = scatter.norm  # Get the normalization from the last scatter
cbar = ColorbarBase(ax_colorbar, cmap=scatter.cmap, norm=norm, orientation='vertical')
cbar.set_label('Fitness against starting deck', fontsize=24)

plt.tight_layout(pad=0)
#plt.savefig(figure_folder + "hearthstone_archive_coverage_against_starting_deck_comparison.pdf")

# %%
for value in ['QD_score', "coverage"]:
    print(value)
    for xp_name in game_xps.keys():
        print(xp_name)
        game_xp = game_xps[xp_name]
        me_xp = me_xps[xp_name]
        for side in ["red", "blue"]:
            print("GAME:", classes_names[game_xp[side]], f"{game_xp[f'{side}_game_against_starting_deck'][value]:2.2f}")
        for side in ["red", "blue"]:
            print("ME:", classes_names[game_xp[side]], f"{me_xp[f'{side}_me'][value]:2.2f}")


# %% [markdown]
# ### ME VS GAME using same grid

# %%
def compute_elites_from_grid(archives, all_B, step_size):
    B = np.array(archives["behavior"]).reshape((1000,2))
    B = normalize_B(B, all_B)
    F = np.array(archives["fitness"]).reshape((1000))
    S = [deck for archive in archives["solutions"] for deck in archive.values()]
    
    cells = {}
    for i in range(len(B)):
        key = tuple([int(x/step_size) if x != 1 else (int(1/step_size)-1) for x in B[i]])  # make sure to have 1/step_size bins per dimension when behavior = max bound
        if key not in cells or F[i] > cells[key]["f"]:
            cells[key] = {"f": F[i], "b": B[i], "s": S[i]}
    return [e["s"] for e in cells.values()]


# %% [raw]
# step_size = 0.1
# for xp_name, xp in game_xps.items():
#     print(xp_name)
#     path = xp["path"]
#     game_xp = game_xps[xp_name]
#     me_xp = me_xps[xp_name]
#
#     red_game_archives = utils.load_pickle(path + f"gen_{6}/archive_save.pk")
#     blue_game_archives = utils.load_pickle(path + f"gen_{7}/archive_save.pk")
#     all_B = np.concatenate( [np.array(red_game_archives["behavior"]).reshape((1000,2)), np.array(blue_game_archives["behavior"]).reshape((1000,2)),
#                              np.array(me_xp[f"blue_me"]["archive"]["behavior"]).reshape((1000,2)), np.array(me_xp[f"red_me"]["archive"]["behavior"]).reshape((1000,2))])
#         
#     game_blues = compute_elites_from_grid(blue_game_archives, all_B, step_size)
#     me_blues = compute_elites_from_grid(me_xp[f"blue_me"]["archive"], all_B, step_size)
#     blues = game_blues + me_blues
#
#     game_reds = compute_elites_from_grid(red_game_archives, all_B, step_size)
#     me_reds = compute_elites_from_grid(me_xp[f"red_me"]["archive"], all_B, step_size)
#     reds = game_reds + me_reds
#     print(f"{len(reds)} = {len(game_reds)} + {len(me_reds)}, {len(blues)} = {len(game_blues)} + {len(me_blues)}")
#     config = get_config(hearthstone_config, 0, None, 0, "red", seed_id=0)
#     tournament = compute_tournament(config, reds, blues)
#     pass # need to update reds and blues 
#     utils.save_pickle(path + f"game_vs_me_tournament_grid.pk", {"tournament": tournament, "reds": me_reds, "blues": me_blues, "step_size": step_size,
#                                                            "game_blues": game_blues, "game_reds": game_reds, })

# %%
for xp_name, xp in game_xps.items():
    path = xp["path"]
    dic = utils.load_pickle(path + f"game_vs_me_tournament_grid.pk")
    n_reds = len(dic["reds"]) + len(dic["game_reds"])
    n_blues = len(dic["blues"]) + len(dic["game_blues"])
    
    tournament_matrix = np.ones((n_reds, n_blues))*0.5
    for (i,j), ev in dic["tournament"].items():
        tournament_matrix[i,j] = (ev["eval"]["fitness"]+30)/60
    red_elo, blue_elo = asymmetric_elo(tournament_matrix)
    dic["F_matrix"] = tournament_matrix
    dic["red_elo"] = red_elo
    dic["blue_elo"] = blue_elo
    xp["GAME_vs_ME_grid"] = dic

# %%
figure_folder = "/home/tim/Experiments/data/GAME/Hearthstone/figures/"

# %%
fig = plt.subplots(figsize=(20,6))
plt.axis("off")
for xp_id, (name, xp) in enumerate(game_xps.items()):
    red_ratings = xp["GAME_vs_ME_grid"]["red_elo"]
    blue_ratings =  xp["GAME_vs_ME_grid"]["blue_elo"]
    
    ax = plt.subplot2grid((2, 5), (0, xp_id))

    n_game_reds = len(xp["GAME_vs_ME_grid"]["game_reds"])
    data = [red_ratings[:n_game_reds], red_ratings[n_game_reds:]]
    names = ["GAME", "ME"]
    plot_boxplot(data, names,  title=f"({['a', 'b', 'c', 'd', 'e'][xp_id]})\n"+classes_names[xp["red"]]+"\n", swarmsize=2, ylabel="ELO Score" if xp_id == 0 else "", fig=fig, ax=ax, colors=["tab:blue", "tab:orange"], use_stick=True, fontsize=18, titlefontsize=18);
    
    ax = plt.subplot2grid((2, 5), (1, xp_id))
    n_game_blues = len(xp["GAME_vs_ME_grid"]["game_blues"])
    data = [blue_ratings[:n_game_blues], blue_ratings[n_game_blues:]]
    plot_boxplot(data, names, swarmsize=2, title=classes_names[xp["blue"]]+"\n", ylabel="ELO Score" if xp_id == 0 else "", fig=fig, ax=ax, colors=["tab:blue", "tab:orange"], cmap="Blues", use_stick=True, fontsize=18, titlefontsize=18);
plt.tight_layout()
plt.savefig(figure_folder + "hearthstone_ELO_comparison.pdf")

# %%
step_size = 0.1

for xp_name in game_xps.keys():
    game_xp = game_xps[xp_name]

    n_game_reds = len(game_xp["GAME_vs_ME_grid"]["game_reds"])
    n_game_blues = len(game_xp["GAME_vs_ME_grid"]["game_blues"])
    
    n_me_reds = len(game_xp["GAME_vs_ME_grid"]["reds"])
    n_me_blues = len(game_xp["GAME_vs_ME_grid"]["blues"])
    
    n_blues = n_game_blues+n_me_blues
    n_reds = n_game_reds+n_me_reds

    red_behaviors = np.zeros((n_reds, 2))
    blue_behaviors = np.zeros((n_blues, 2))
    good_reds = np.zeros((n_reds))
    good_blues = np.zeros((n_blues))
    
    for (i,j), ev in game_xp["GAME_vs_ME_grid"]["tournament"].items():
        if i not in red_behaviors:
            red_behaviors[i] = ev["eval"]["behavior"]
            good_reds[i] = True
        if j not in blue_behaviors:
            blue_behaviors[j] = ev["eval"]["other_behavior"]
            good_blues[j] = True

    all_B = np.concatenate([red_behaviors[np.where(good_reds)], blue_behaviors[np.where(good_blues)]])
    
    game_xp["GAME_vs_ME_grid"]["all_B"] = all_B
    
    for variant in ["GAME", "ME"]:
        if variant == "GAME":
            red_ids = np.where(good_reds[:n_game_reds])[0]
            blue_ids = np.where(good_blues[:n_game_reds])[0]
        else:
            red_ids = n_game_reds + np.where(good_reds[n_game_reds:])[0]
            blue_ids = n_game_reds + np.where(good_blues[n_game_reds:])[0]
        game_xp["GAME_vs_ME_grid"][variant] = {}
        
        for side in ["red", "blue"]:
            dic = {}
            if side == "red":
                B = red_behaviors[red_ids]
                F = np.mean(game_xp["GAME_vs_ME_grid"]["F_matrix"][red_ids][:, np.where(good_blues)[0]], axis=1)
            else:
                B = blue_behaviors[blue_ids]
                F = 1-np.mean(game_xp["GAME_vs_ME_grid"]["F_matrix"][np.where(good_reds)[0]][:, blue_ids], axis=0)
            qd, cov, cells = compute_qd_score(normalize_B(B, all_B), np.array(F), step_size)
            dic["QD_score"] = qd[-1]
            dic["coverage"] = cov[-1]
            dic["cells"] = cells
            game_xp["GAME_vs_ME_grid"][variant][side] = dic 

# %%
n = int(1/step_size)
n_sticks = 5

# Create one big figure with 3x2 outer grid, each containing 2x2 inner subplots
fig = plt.figure(figsize=(18, 12))  # Wide figure to accommodate 3 columns

exp_names = list(game_xps.keys())
n_experiments = len(exp_names)

for exp_idx, xp_name in enumerate(exp_names):
    game_xp = game_xps[xp_name]
    
    all_B = game_xp["GAME_vs_ME_grid"]["all_B"]
    B_m, B_M = get_bounds(all_B)
    
    # Calculate outer grid position (3 columns, 2 rows)
    outer_row = exp_idx // 3
    outer_col = exp_idx % 3
    
    for i, side in enumerate(["red", "blue"]):
        for j, variant in enumerate(["GAME", "ME"]):
            # Each outer cell gets a 2x2 grid within it
            # Grid calculation: (outer_rows * inner_rows, outer_cols * inner_cols)
            # Position: (outer_row * inner_rows + inner_row, outer_col * inner_cols + inner_col)
            subplot_row = outer_row * 2 + j
            subplot_col = outer_col * 2 + i
            
            ax = plt.subplot2grid((4, 6), (subplot_row, subplot_col))  # 4 rows (2*2), 6 cols (3*2)
            
            cells = game_xp["GAME_vs_ME_grid"][variant][side]["cells"]
            M = np.zeros((n,n))
            for (x,y), f in cells.items():
                M[x,y] = f
            im = ax.pcolor(M.T, vmin=0, vmax=1)
            
            x = np.linspace(0, 1, n_sticks)
            ax.set_xticks(x*n)
            ax.set_xticklabels([f'{v:1.1f}' for v in x*(B_M[0]-B_m[0])+B_m[0]], fontsize=8)
            ax.set_yticks(x*n)
            ax.set_yticklabels([f'{v:1.1f}' for v in x*(B_M[1]-B_m[1])+B_m[1]], fontsize=8)
            

            title = f"{classes_names[game_xp[side]]} - QD:{game_xp['GAME_vs_ME_grid'][variant][side]['QD_score']:2.2f}"
            ax.set_title(title, fontsize=10)
            if j!= 0:  # Bottom row
                ax.set_xlabel("Mana cost mean", fontsize=9)
            if i == 0:  # Left column of each 2x2
                ax.set_ylabel(f"{variant}\nMana cost variation", fontsize=9)
            


plt.tight_layout()
fig.subplots_adjust(right=0.94)  # Make space for colorbar
cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Fitness', fontsize=12)

plt.show()


# %% [markdown]
# ### ME VS GAME (using k-means)

# %%
def get_deck_key(solution):
    l = list(solution["deck"])
    l.sort()
    return "_".join(l)

def compute_elites(archives, k):
    B = np.array(archives["behavior"]).reshape((1000,2))
    F = np.array(archives["fitness"]).reshape((1000))
    S = [deck for archive in archives["solutions"] for deck in archive.values()]
    
    kmeans = KMeans(n_clusters=k).fit(B)
    centroids = kmeans.cluster_centers_
    tree = cKDTree(centroids)
    
    elites = [{"fitness": None, "solution": None, "behavior": None} for _ in range(k)]
    elites_solution_txt = [""] * k
    distances_to_centroids = np.ones(k) * np.inf
    
    for s, f, b in zip(S, F, B):
        distance, c_id = tree.query(b, k=1)
        if compare_fitness(f, elites[c_id]["fitness"]) and get_deck_key(s) not in elites_solution_txt:
            elites[c_id]["fitness"] = f
            elites[c_id]["solution"] = s
            elites[c_id]["behavior"] = b
            elites_solution_txt[c_id] = get_deck_key(s)
    b = [e["behavior"] for e in elites]
    return [e["solution"] for e in elites]


# %%
k = 50

# %% [raw]
# for xp_name, xp in game_xps.items():
#     print(xp_name)
#     path = xp["path"]
#     game_xp = game_xps[xp_name]
#     me_xp = me_xps[xp_name]
#     blues, reds = [], []
#     for (side, gen_id) in [("blue", 7), ("red", 6)]:
#         archives = utils.load_pickle(path + f"gen_{gen_id}/archive_save.pk")
#         if side == "blue":
#             blues = compute_elites(archives, k) + compute_elites(me_xp[f"{side}_me"]["archive"], k)
#         else:
#             reds = compute_elites(archives, k) + compute_elites(me_xp[f"{side}_me"]["archive"], k)
#     assert len(blues) == k*2 and len(reds) == k*2
#     config = get_config(hearthstone_config, 0, None, 0, "red", seed_id=0)
#     tournament = compute_tournament(config, reds, blues)
#     utils.save_pickle(path + f"game_vs_me_tournament.pk", {"tournament": tournament, "reds": reds, "blues": blues})

# %%
for xp_name, xp in game_xps.items():
    path = xp["path"]
    dic = utils.load_pickle(path + f"game_vs_me_tournament.pk")
    tournament_matrix = np.ones((k*2, k*2))*0.5
    for (i,j), ev in dic["tournament"].items():
        tournament_matrix[i,j] = (ev["eval"]["fitness"]+30)/60
    red_elo, blue_elo = asymmetric_elo(tournament_matrix)
    dic["F_matrix"] = tournament_matrix
    dic["red_elo"] = red_elo
    dic["blue_elo"] = blue_elo
    xp["GAME_vs_ME"] = dic

# %% [markdown]
# #### ELO score plot

# %%
fig = plt.subplots(figsize=(20,6))
plt.axis("off")
for xp_id, (name, xp) in enumerate(game_xps.items()):
    red_ratings = xp["GAME_vs_ME"]["red_elo"]
    blue_ratings =  xp["GAME_vs_ME"]["blue_elo"]
    
    ax = plt.subplot2grid((2, 5), (0, xp_id))
    n_tasks = 50
    n_gen = 8
    data = [red_ratings[:k], red_ratings[k:]]
    names = ["GAME", "ME"]
    plot_boxplot(data, names,  swarmsize=2, ylabel=classes_names[xp["red"]]+"\nELO Score", fig=fig, ax=ax, colors=["gold", "royalblue"], use_stick=True);
    
    ax = plt.subplot2grid((2, 5), (1, xp_id))
    data = [blue_ratings[:k], blue_ratings[k:]]
    plot_boxplot(data, names, swarmsize=2, ylabel=classes_names[xp["blue"]]+"\nELO Score", fig=fig, ax=ax, colors=["gold", "royalblue"], cmap="Blues", use_stick=True);

# %% [markdown]
# #### QD-Score plot (with shared grid archive)

# %%
step_size = 1/7

for xp_name in game_xps.keys():
    game_xp = game_xps[xp_name]
    me_xp = me_xps[xp_name]
    
    red_behaviors = np.zeros((100,2))
    blue_behaviors = np.zeros((100,2))
    good_reds = np.zeros((100))
    good_blues = np.zeros((100))
    for (i,j), ev in game_xp["GAME_vs_ME"]["tournament"].items():
        if i not in red_behaviors:
            red_behaviors[i] = ev["eval"]["behavior"]
            good_reds[i] = True
        if j not in blue_behaviors:
            blue_behaviors[j] = ev["eval"]["other_behavior"]
            good_blues[j] = True

    all_B = np.concatenate([red_behaviors[np.where(good_reds)], blue_behaviors[np.where(good_blues)]])
    
    game_xp["GAME_vs_ME"]["all_B"] = all_B
    
    for variant in ["GAME", "ME"]:
        if variant == "GAME":
            red_ids = np.where(good_reds[:k])[0]
            blue_ids = np.where(good_blues[:k])[0]
        else:
            red_ids = k + np.where(good_reds[k:])[0]
            blue_ids = k + np.where(good_blues[k:])[0]
        game_xp["GAME_vs_ME"][variant] = {}
        dic = {}
        for side in ["red", "blue"]:
            if side == "red":
                B = red_behaviors[red_ids]
                F = np.mean(game_xp["GAME_vs_ME"]["F_matrix"][red_ids][:, np.where(good_blues)[0]], axis=1)
            else:
                B = blue_behaviors[blue_ids]
                F = 1-np.mean(game_xp["GAME_vs_ME"]["F_matrix"][np.where(good_reds)[0]][:, blue_ids], axis=0)
            qd, cov, cells = compute_qd_score(normalize_B(B, all_B), np.array(F), step_size)
            dic["QD_score"] = qd[-1]
            dic["coverage"] = cov[-1]
            dic["cells"] = cells
            game_xp["GAME_vs_ME"][variant][side] = dic 

# %%
n = int(1/step_size)
n_sticks = 5

# Create one big figure with 3x2 outer grid, each containing 2x2 inner subplots
fig = plt.figure(figsize=(18, 12))  # Wide figure to accommodate 3 columns

exp_names = list(game_xps.keys())
n_experiments = len(exp_names)

for exp_idx, xp_name in enumerate(exp_names):
    game_xp = game_xps[xp_name]
    
    all_B = game_xp["GAME_vs_ME"]["all_B"]
    B_m, B_M = get_bounds(all_B)
    
    # Calculate outer grid position (3 columns, 2 rows)
    outer_row = exp_idx // 3
    outer_col = exp_idx % 3
    
    for i, side in enumerate(["red", "blue"]):
        for j, variant in enumerate(["GAME", "ME"]):
            # Each outer cell gets a 2x2 grid within it
            # Grid calculation: (outer_rows * inner_rows, outer_cols * inner_cols)
            # Position: (outer_row * inner_rows + inner_row, outer_col * inner_cols + inner_col)
            subplot_row = outer_row * 2 + j
            subplot_col = outer_col * 2 + i
            
            ax = plt.subplot2grid((4, 6), (subplot_row, subplot_col))  # 4 rows (2*2), 6 cols (3*2)
            
            cells = game_xp["GAME_vs_ME"][variant][side]["cells"]
            M = np.zeros((n,n))
            for (x,y), f in cells.items():
                M[x,y] = f
            im = ax.pcolor(M.T, vmin=0, vmax=1)
            
            x = np.linspace(0, 1, n_sticks)
            ax.set_xticks(x*n)
            ax.set_xticklabels([f'{v:1.1f}' for v in x*(B_M[0]-B_m[0])+B_m[0]], fontsize=8)
            ax.set_yticks(x*n)
            ax.set_yticklabels([f'{v:1.1f}' for v in x*(B_M[1]-B_m[1])+B_m[1]], fontsize=8)
            

            title = f"{classes_names[game_xp[side]]} - QD:{game_xp['GAME_vs_ME'][variant][side]['QD_score']:2.2f}"
            ax.set_title(title, fontsize=10)
            if j!= 0:  # Bottom row
                ax.set_xlabel("Mana cost mean", fontsize=9)
            if i == 0:  # Left column of each 2x2
                ax.set_ylabel(f"{variant}\nMana cost variation", fontsize=9)
            


plt.tight_layout()
fig.subplots_adjust(right=0.94)  # Make space for colorbar
cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Fitness', fontsize=12)

plt.show()

# %% [markdown]
# #### QD-score Plot (not real QD because use different archives)

# %%
for xp_name, xp in game_xps.items():
    red_behaviors = np.zeros((100,2))
    blue_behaviors = np.zeros((100,2))
    good_reds = np.zeros((100))
    good_blues = np.zeros((100))
    for (i,j), ev in xp["GAME_vs_ME"]["tournament"].items():
        if i not in red_behaviors:
            red_behaviors[i] = ev["eval"]["behavior"]
            good_reds[i] = True
        if j not in blue_behaviors:
            blue_behaviors[j] = ev["eval"]["other_behavior"]
            good_blues[i] = True

    for variant in ["GAME", "ME"]:
        dic = {}
        if variant == "GAME":
            red_ids = np.where(good_reds[:k])[0]
            blue_ids = np.where(good_blues[:k])[0]
        else:
            red_ids = k + np.where(good_reds[k:])[0]
            blue_ids = k + np.where(good_blues[k:])[0]
        dic["red_B"] = red_behaviors[red_ids]
        dic["red_F"] = np.mean(xp["GAME_vs_ME"]["F_matrix"][red_ids][:, np.where(good_blues)[0]], axis=1)
        dic["blue_B"] = blue_behaviors[blue_ids]
        dic["blue_F"] = 1-np.mean(xp["GAME_vs_ME"]["F_matrix"][np.where(good_reds)[0]][:, blue_ids], axis=0)
        xp["GAME_vs_ME"][variant] = dic 
    dic = game_xp

# %%
for xp_name, game_xp in game_xps.items():
    dic = game_xp["GAME_vs_ME"]

    plt.subplots(figsize=(4*3, 4*2))
    plt.axis("off")
    
    all_B = np.concatenate([dic["ME"]["red_B"], dic["ME"]["blue_B"],dic["GAME"]["red_B"], dic["GAME"]["blue_B"]])
    xmin, ymin = np.min(all_B, axis=0)
    xmax, ymax = np.max(all_B, axis=0)
    xpad = (xmax-xmin)*0.05
    ypad = (ymax-ymin)*0.05
    
    for i, side in enumerate(["red", "blue"]):
        for j, variant in enumerate(["GAME", "ME"]):
            plt.subplot2grid((2, 21), (j, 10*i), colspan=10)
            B = dic[variant][f"{side}_B"]
            F = dic[variant][f"{side}_F"]
            scatter = plt.scatter(B[:,0], B[:,1], s=20, c=F, vmin=0, vmax=1, alpha=1, cmap="viridis")
            plt.xlim((xmin-xpad, xmax+xpad))
            plt.ylim((ymin-ypad, ymax+ypad))
            print(xp_name, classes_names[game_xp[side]], variant, f"{np.mean(F):2.2f}")
            if j == 0:
                plt.title(classes_names[game_xp[side]])
            else:
                plt.xlabel("Mana cost mean")
            if i == 0:
                plt.ylabel(f"{variant}\nMana cost variation")
    
    ax_colorbar = plt.subplot2grid((2, 21), (0, 20), rowspan=2, colspan=1)
    norm = scatter.norm  # Get the normalization from the scatter
    cbar = ColorbarBase(ax_colorbar, cmap=scatter.cmap, norm=norm, orientation='vertical')
    cbar.set_label('Fitness', fontsize=24)
