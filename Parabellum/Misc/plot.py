# # Config

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

font = {'size'   : 18}
mpl.rc('font', **font)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
#plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"] = (16, 9)
plt.rcParams["figure.facecolor"] = "White"

mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = False

# +
blue = '#4b4dce'
green = '#117733'
light_green = '#44AA99'
light_blue = '#88CCEE'
yellow = '#DDCC77'
red = "#CC6677"


LightGreen = '#90EE90'
Teal = '#008080'

Pink = '#FFC0CB'

LightBlue ='#ADD8E6'
Purple = '#800080'
DarkBlue = '#000080'
Yellow = '#DDCC77'#'#FFFF00'
Brown = '#A52A2A'
Gray = '#808080'
Orange = '#FFA500'
grad = [blue, light_blue, light_green, green, yellow, red, '#AA4499', "#000000"]
colors = [light_green, red, light_blue, green, blue, yellow, '#AA4499', "#000000"]*10


# -

def int_to_color(i, vmax, vmin=0, cmap=cm.rainbow, in_int=False):
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.ScalarMappable(norm=norm, cmap=cmap)
    c = cmap.to_rgba(i)
    if in_int:
        c = tuple(np.array(np.array(c) * 255, dtype=int) )
    return c


linestyles = [
       (0, (1, 1)), #    ('dotted',  
       (0, (5, 5)),  #      ('dashed',             
       (0, (3, 5, 1, 5)),  #      ('dashdotted',           
       (0, (3, 5, 1, 5, 1, 5)),  #     ('dashdotdotted',      
       (5, (10, 3)),  #      ('long dash with offset', 
    
       (0, (5, 1)),  #     ('densely dashed',     
       (0, (3, 1, 1, 1)),  #      ('densely dashdotted',  
       (0, (3, 1, 1, 1, 1, 1)),  #      ('densely dashdotdotted', 

       (0, (1, 10)), #      ('loosely dotted',        
       (0, (5, 10)),  #      ('loosely dashed',       
       (0, (3, 10, 1, 10)),  #      ('loosely dashdotted',    
       (0, (3, 10, 1, 10, 1, 10)),  #      ('loosely dashdotdotted',
] 



# # Miscaleneous 

def sort_names_and_data(names, data):
    tups = list(zip(names, data))
    tups.sort()
    return zip(*tups)


def enforce_newline(s, n):
    words = s.split()
    result = []
    current_line = ""

    for word in words:
        # Check if adding the next word exceeds the length limit
        if len(current_line) + len(word) + 1 > n:
            # If the current line is not empty, add it to the result
            if current_line:
                result.append(current_line)
                current_line = ""
        # Add the word to the current line
        if current_line:
            current_line += " " + word
        else:
            current_line = word
    
    # Add the last line to the result if it's not empty
    if current_line:
        result.append(current_line)
    
    return "\n".join(result)


# # Boxplot

# +
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


# -

def plot_boxplot(data, names, ylabel="performance", ylim=None, title="", log=False, bbox=(1.13,0.1,0.5,0.9), cmap=cm.rainbow, figsize=(16,9), fig=None, ax=None,
                 correction=True, rotation=0, use_table=False, use_stick=False, swarmsize=7, force_swarm=False, swarmdata=None, colors=None, verticale=True, x_ticks=True, y_ticks=True, xlabel=None):
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
        plt.text(s=stat[0][2], x=0.5, y=toptop, ha="center")
    if x_ticks:
        (plt.xticks if verticale else plt.yticks)(range(N), names, rotation=rotation)
    else:
        (plt.xticks if verticale else plt.yticks)([])
    if log:
        plt.yscale('log')
    if not ylim is None:
        plt.ylim(ylim)
    if not y_ticks:
        (plt.yticks if verticale else plt.xticks)([])
    if ylabel:
        plt.ylabel(ylabel) if verticale else plt.xlabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel) if verticale else plt.ylabel(xlabel)
    plt.title(title)
    txt += "\n"
    for i in range(N):
        txt += f"{names[i]}: {np.median(data[i]):2.1f} [{np.quantile(data[i], 0.25)}, {np.quantile(data[i], 0.75)}]\n"
    txt += "\n"
    for i in range(N):
        txt += f"{names[i]}: {np.median(data[i]):2.1f} [{np.quantile(data[i], 0.25):2.1f}, {np.quantile(data[i], 0.75):2.1f}]\n"
    return txt 


# # plot_with_spread

def plot_with_spread(data, names, X, colors=cm.rainbow, lw=3, alpha=0.3, line_styles=None, plot_fst_variance=True, plot_snd_variance=False):
    """
    dim0: different methods
    dim1: replication 
    dim2: abscice 
    """
    line_styles = ["-"] * len(data) if line_styles is None else line_styles
    for i, Y in enumerate(data):
        median = np.median(Y, axis=0) 
        q_05 = np.quantile(Y, 0.05, axis=0) 
        q_25 = np.quantile(Y, 0.25, axis=0) 
        q_75 = np.quantile(Y, 0.75, axis=0) 
        q_95 = np.quantile(Y, 0.95, axis=0) 
        color = colors[i] if type(colors) == list else int_to_color(i, len(data), cmap=colors)
        x = X[i]
        plt.plot(x, median, lw=lw, color=color, ls=line_styles[i], label=names[i])
        if plot_fst_variance:
            plt.fill_between(x, q_25, q_75 , color=color, alpha=alpha)
        if plot_snd_variance:
            plt.fill_between(x, q_05, q_95 , color=color, alpha=0.3*alpha)
    if len(data) > 1:
        plt.legend()

# + active=""
# X = [[i for i in range(100)]]* 2
# names = ["A", "B"]
# data = np.random.random((2, 3, 100))
# plot_with_spread(data, names, X)

# + active=""
# data = list(np.random.random((3,100))) + list(np.random.random((1,100))*2)
# names = ["1", "2", "3", "4"]
# plot_boxplot(data,names, use_table=True)

# + active=""
# data = list([np.random.random((100))*(i+1) for i in range(2)])
# names = ["z", "a"]
# names, data = sort_names_and_data(names, data)
# plot_boxplot(data, names, use_table=False, use_stick=True)
# -


