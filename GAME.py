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

# %% [raw]
# import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# %%
import sys
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

# %%
sys.path.append("Misc") 
import utils
from plot import *
import plot
from grid_plot import * 
sys.path.append("Parabellum")  # import utils from parent folder 
from BT_tree import BT, txt2tree
from eval_bi_bts import eval_bi_bts_factory
import grammar

# %%
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # for the video running while JAX is still in use (doesn't like it)

# %%
N_PROC = 1


# %% [markdown]
# # Functions 

# %% [markdown]
# ## Growing Archive

# %%
def cvt(k, dim, coef=10, verbose=False, rep=0):
    root = "../../data/cvt/"
    name = f"{int(k)}_{int(dim)}_{rep}.pk"
   
    if os.path.exists(root+name):
        with open(root+name, "rb") as f:
            X = pickle.load(f)
    else:
        rng = np.random.default_rng(rep)
        x = rng.random((k*coef, dim))
        k_means = KMeans(init='k-means++', n_clusters=k, n_init=1, verbose=False)#,algorithm="full")
        k_means.fit(x)
        X = k_means.cluster_centers_
        with open(root+name, "wb") as f:
            pickle.dump(X, f)
    return X


# %%
def _compute_distance(distance_function, b, centroids):
    distances = distance_function(b, centroids)
    c_id = np.argmin(distances)
    return distances[c_id], c_id

def compute_distances(distance_function, centroids):
    n = len(centroids)
    distances = np.zeros((n, n))
    for i in range(n):
        distances[i] = distance_function(centroids[i], centroids)
    return distances

def _compute_neighbors(distance_function, centroids):
    distances = compute_distances(distance_function, centroids)
    c_ids = np.argsort(distances, axis=1)
    distances = np.array([distances[i][c_ids[i]] for i in range(len(centroids))])
    return distances, c_ids


# %%
class GA():
    def __init__(self, config, rng, seed):
        self.rng = rng
        self.seed = seed
        self.use_growing = config["use_growing"]
        self.n_cells = config["n_cells"]
        self.n_solution_dim = config["n_solution_dim"]
        self.n_behavior_dim = config["n_behavior_dim"]
        self.use_redristribution = config["use_redristribution"]
        self.use_collection = config["use_collection"]
        self.use_repair = config["use_repair"]
        self.compute_distance = partial(_compute_distance, config["distance_function"])
        self.compute_neighbors = partial(_compute_neighbors, config["distance_function"])
        self.compare_fitness = config["compare_fitness"]
        self.cells_fitness = [None for _ in range(self.n_cells)]
        self.cells_solution = {} if self.n_solution_dim is None else np.empty(shape=(self.n_cells, self.n_solution_dim)) 
        self.cells_behavior = np.empty(shape=(self.n_cells, self.n_behavior_dim))
        self.cells_log_id = np.ones(shape=(self.n_cells), dtype=np.int32) * -1
        self.cells_former_elites = {i: [] for i in range(self.n_cells)}
        self.non_empty_cells = []
    
        if self.use_growing:
            self.centroids = np.empty(shape=(self.n_cells, self.n_behavior_dim))
        else:
            self.centroids = cvt(self.n_cells, self.n_behavior_dim, rep=self.seed)
        self.n_centroids = 0 if self.use_growing else self.n_cells  
        self.dmin = None
        
    def n_elites(self):
        return len(self.non_empty_cells)

    def sample_parent(self):
        p = self.rng.choice(self.non_empty_cells)
        return self.cells_solution[p], p

    def fill_empty_cell(self, evaluation):
        cell_id = np.where(np.linalg.norm(self.centroids[:self.n_centroids] - evaluation["behavior"], axis=1) == 0)[0]
        if len(cell_id) == 0:  # if not already present creates a new cell
            self.centroids[self.n_centroids] = evaluation["behavior"]
            self.cells_former_elites[self.n_centroids] = [evaluation]
            cell_id = self.n_centroids
            self.n_centroids += 1
            self.non_empty_cells.append(cell_id)
        else:
            cell_id = cell_id[0]
        return cell_id
        
    def set_new_elite(self, cell_id, evaluation, append=True, reset=True):
        self.cells_log_id[cell_id] = evaluation["id"]
        self.cells_fitness[cell_id] = evaluation["fitness"]
        self.cells_solution[cell_id] = evaluation["solution"]
        self.cells_behavior[cell_id] = evaluation["behavior"]
        if append:
            self.cells_former_elites[cell_id].append(deepcopy(evaluation))
        elif reset:
            self.cells_former_elites[cell_id] = [deepcopy(evaluation)]

    def compute_dmin(self):
        self.d_neighbors, self.c_id_neighbors = self.compute_neighbors(self.centroids)
        self.dmin = np.min(self.d_neighbors[:, 1])

    def apply_collection(self, pruned):
        """ collect elites from neighbors """ 
        for j in range(self.n_cells): 
            if len(self.cells_former_elites[j]) > 1 and j != pruned: 
                new_cell_elites, new_elite = [], None
                for elite in self.cells_former_elites[j]:
                    _, elite_id = self.compute_distance(elite["behavior"], self.centroids)
                    if elite_id == pruned: # split the elites between the old and new cells
                        self.cells_former_elites[pruned].append(elite)
                        if self.compare_fitness(elite["fitness"], self.cells_fitness[pruned]): # bootstrap the new cell with already found elites from neighbors
                            self.set_new_elite(pruned, elite, reset=False, append=False)
                    elif j == elite_id:  # repair the archive at the new cell location (neighbor cells can loose their elites) 
                        new_cell_elites.append(elite)
                        if new_elite is None or self.compare_fitness(elite["fitness"], new_elite["fitness"]):
                            new_elite = elite 
                    else:
                        print("Should not go here!")
                self.set_new_elite(j, new_elite, reset=False, append=False)
                self.cells_former_elites[j] = new_cell_elites
                
    def apply_repair(self, pruned):
        """ soft sub-optimal repair in case the current elite of cell is stolen by the new one """
        for j in range(self.n_cells): 
            if len(self.cells_former_elites[j]) > 1 and j != pruned: 
                _, elite_id = self.compute_distance(self.cells_former_elites[j][-1]["behavior"], self.centroids)
                if elite_id == pruned:  # if the new cell steals the current elite we reinstate the initial centroid  
                    self.set_new_elite(j, self.cells_former_elites[j][0], reset=True, append=False)
                    
    def apply_redristribution(self, former_elites):
        """ distribute the pruned elites into their new cells """
        for elite in former_elites:
            _, cell_id = self.compute_distance(elite["behavior"], self.centroids)
            if self.compare_fitness(elite["fitness"], self.cells_fitness[cell_id]):
                self.set_new_elite(cell_id, elite)
    
    def add_evaluation(self, evaluation):
        changed = False
        if not self.use_growing:  # classic MAP-Elites 
            d, cell_id = self.compute_distance(evaluation["behavior"], self.centroids)
        else:  # GAME
            if self.n_centroids < self.n_cells:  # initialisation of the n_cells with the first n different behaviors
                cell_id = self.fill_empty_cell(evaluation)
                if self.n_centroids == self.n_cells:  # the archive is full, we can compute the minimal distance 
                    self.compute_dmin()
            else:  # only grows if the new solution is farther than the closest two cells
                d, cell_id = self.compute_distance(evaluation["behavior"], self.centroids)
                if d > self.dmin:
                    centroid_A = np.argmin(self.d_neighbors[:, 1])
                    centroid_B = self.c_id_neighbors[centroid_A, 1]
                    pruned = centroid_A if self.d_neighbors[centroid_A, 2] < self.d_neighbors[centroid_B, 2] else centroid_B
                    self.centroids[pruned] = evaluation["behavior"]
                    self.compute_dmin()
                    former_elites = deepcopy(self.cells_former_elites[pruned])
                    changed = True 
                    self.set_new_elite(pruned, evaluation, reset=True, append=False)
                    if self.use_redristribution: 
                        self.apply_redristribution(former_elites)
                    if not self.use_collection and self.use_repair:  
                        self.apply_repair(pruned) 
                    if self.use_collection:  
                        self.apply_collection(pruned)
    
        if self.compare_fitness(evaluation["fitness"], self.cells_fitness[cell_id]) and not changed:
            self.set_new_elite(cell_id, evaluation)
            if not self.use_growing and cell_id not in self.non_empty_cells:  # keeps track of the filled cells for ME 
                self.non_empty_cells.append(cell_id)


# %% [markdown]
# ## Multi-Task Archive

# %%
class MTGA():
    def __init__(self, config, n_tasks, rng, seed):
        self.rng = rng 
        self.n_tasks = n_tasks
        self.archives = [GA(config, self.rng, seed) for _ in range(self.n_tasks)]
        self.non_empty_archive = []
        
    def update(self, evaluations):
        for evaluation in evaluations:
            self.archives[evaluation['task_id']].add_evaluation(evaluation)
            if evaluation['task_id'] not in self.non_empty_archive:
                self.non_empty_archive.append(evaluation['task_id'])
        
    def sample_parents(self):
        p1_a_id, p2_a_id = self.rng.choice(self.non_empty_archive, 2)
        p1, p1_id = self.archives[p1_a_id].sample_parent()
        p2, p2_id = self.archives[p2_a_id].sample_parent()
        return p1, p2, (p1_a_id, p1), (p2_a_id, p2)

    def n_elites(self):
        return np.sum([archive.n_elites() for archive in self.archives])


# %% [markdown]
# ## Mt GAME

# %%
class MT_GAME():
    def __init__(self, config):
        self.config = config 
        self.rng = np.random.default_rng(config["seed"])
        self.tasks = config["tasks"]
        self.n_tasks = len(self.tasks)
        self.archive = MTGA(config["archive_config"], self.n_tasks, self.rng, config["seed"])
        self.evaluation_function = partial(config["evaluation_function"], **config["evaluation_config"])
        self.sample_random = config["sample_random_function"]
        self.sample_crossover_and_mutation = config["crossover_and_mutation_function"]
        self.n_proc = config["n_proc"]
        self.budget = config["budget"]
        self.batch_size = config["batch_size"]  # number of individuals evaluated in // on the same task 
        self.parallel = config["parallel"]
        self.verbose = config["verbose"]
        self.save_folder = config["save_folder"]
        self.log_interval = config["log_interval"]
        self.init_elites = config["init_elites"]
        self.avoid_repetition = config["avoid_repetition"] and (config["archive_config"]["n_solution_dim"] is None)  # only for BTs which are discrete
        self.alpha = config["alpha"] if "alpha" in config else None  # scale of the behavior space for the benchmark comparison 

        self.intermediate_videos = config["intermediate_videos"]
        self.first_video = config["first_video"]
        self.last_video = config["last_video"]
        self.make_videos_function = partial(make_videos, config["video_eval_fn"])
        self.already_tried_bts = set()
        self.rejected = 0
        self.is_random = True 
        self.it_end_random = ""
        self.it = 0
        self.log = []

    def __getstate__(self):
        # Exclude non-serializable resources
        state = self.__dict__.copy()
        del state['evaluation_function']
        del state["make_videos_function"]
        del state["config"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize non-serializable resources, if necessary
        self.evaluation_function = None
        
    def save_archive(self):
        archive_save = {"fitness": [], "solutions": [], "behavior": [], "log_id": []}
        for archive in self.archive.archives:
            archive_save["fitness"].append(archive.cells_fitness)
            archive_save["solutions"].append(archive.cells_solution)
            archive_save["behavior"].append(archive.cells_behavior)
            archive_save["log_id"].append(archive.cells_log_id)
        archive_save["log"] = self.log
        archive_save["batch_size"] = self.batch_size
        utils.save_pickle(self.save_folder+"/archive_save.pk", archive_save) 
        
    def sample_candidate(self):
        if self.is_random:  # initialization with random solutions till we find enough elites for diversity
            candidate, origin = self.sample_random(self.rng, config["random_sampling_config"])
        else:               
            p1, p2, p1_id, p2_id = self.archive.sample_parents()
            candidate, origin = self.sample_crossover_and_mutation(self.rng, p1, p2, config["crossover_and_mutation_config"])
            origin["p1"] = p1_id
            origin["p2"] = p2_id
        return {"value": candidate, "origin": origin}

    def sample_task(self):
        task_id = self.rng.integers(self.n_tasks)
        return {"config": self.tasks[task_id], "id": task_id}

    def sample_tasks(self):
        indexes = self.rng.choice(np.arange(self.n_tasks), self.batch_size, replace=self.batch_size > self.n_tasks)
        return [{"config": self.tasks[task_id], "id": task_id} for task_id in indexes]
    
    def sample_new_evaluations(self):
        tasks = self.sample_tasks()
        candidates = [self.sample_candidate() for _ in range(self.batch_size)]
        return tasks, candidates
        
    def update_archive(self, evaluations):
        has_changed = False
        for ev in evaluations:
            ev["id"] = self.it
            self.it += 1
        self.archive.update(evaluations)
        self.logger(evaluations)
        if self.is_random and self.archive.n_elites() >= self.init_elites:
            self.is_random = False 
            self.it_end_random = self.it 
            has_changed = True
        return has_changed

    def logger(self, evaluations):
        self.log.append(evaluations)

    def get_fitness(self):
        return [ev["fitness"] for batch in self.log for ev in batch]

    def get_behaviors(self):
        return [ev["behavior"] for batch in self.log for ev in batch]

    def get_solutions(self):
        return [ev["solution"] for batch in self.log for ev in batch]

    def get_descriptions(self):
        return [ev["description"] for batch in self.log for ev in batch]
    
    def compute_qd_score(self, behaviors, fitness, step_size=0.01, alpha=1., verbose=False):
        qd_score, coverage = np.empty(len(behaviors)), np.empty(len(behaviors))
        cells = {}
        for i in tqdm(range(len(behaviors))) if verbose else range(len(behaviors)):
            key = tuple([int(x) for x in (behaviors[i]/alpha+(alpha-1)/(2*alpha))/step_size])
            if key not in cells:
                cells[key] = fitness[i]
            else:
                cells[key] = max(cells[key], fitness[i])
            qd_score[i] = sum((x for x in cells.values())) * step_size**len(key)
            coverage[i] = len(cells)* step_size**len(key)
        return qd_score, coverage, cells
    
    def final_log(self):
        if self.archive.archives[0].n_behavior_dim == 2 and self.alpha is not None:
            qd_score, coverage, cells = self.compute_qd_score(self.get_behaviors(), self.get_fitness(), alpha=self.alpha)
            self.dq_score = qd_score
            self.coverage = coverage
            self.cells = cells 

    def run(self):
        if self.parallel:
            job_queue = mp.Queue()
            res_queue = mp.Queue()
            pool = mp.Pool(self.n_proc, parallel_worker, (self.evaluation_function, job_queue, res_queue))
            for _ in range(self.n_proc):
                tasks, candidates = self.sample_new_evaluations()
                job_queue.put({"tasks": tasks, "candidates": candidates})
        else:
            job_queue = [self.sample_new_evaluations()]
            
        if self.verbose > 0: # create loading bar info 
            loading_bar = tqdm(total=self.budget-self.it, ncols=100, smoothing=0.01, mininterval=1) 
            loading_bar.set_description("Random" if self.is_random else f"C&M [{self.it_end_random}]")
            
        for it in range(self.it//self.batch_size, self.budget//self.batch_size):
            if self.parallel:  # collect the evaluation
                evaluated_candidates = res_queue.get()
            else:
                tasks, candidates = job_queue.pop(0)
                evaluated_candidates = self.evaluation_function(tasks=tasks, candidates=candidates)
                
            has_changed = self.update_archive(evaluated_candidates)
                                    
            if it*self.batch_size % self.log_interval == 0:  # save archive 
                #utils.save_pickle(self.save_folder + "/xp.pk", self)
                self.save_archive()
                if self.archive.n_elites() == self.archive.archives[0].n_cells and self.intermediate_videos:
                    make_videos(str(it*self.batch_size), self, parabellum_config["n_steps"])
            
            if self.verbose:  # update loading bar info
                loading_bar.update(self.batch_size)
                if has_changed:
                    loading_bar.set_description(f"C&M [{self.it}]")
                    if self.first_video:
                        make_videos("first", self, parabellum_config["n_steps"])
           
            tasks, candidates = self.sample_new_evaluations()                
            if self.parallel:  # put a new job  
                job_queue.put({"tasks": tasks, "candidates": candidates})
            else:
                job_queue.append((tasks, candidates))
            
        if self.verbose:
            loading_bar.close()

        if self.parallel:
            job_queue.close()
            res_queue.close()
            pool.terminate()
         
        #utils.save_pickle(self.save_folder+"/xp.pk", self) 
        self.save_archive()
        if self.last_video:
            make_videos("final", self, parabellum_config["n_steps"])


# %% [markdown]
# ## Miscaleanous

# %% [markdown]
# ### workers 

# %%
def parallel_worker(evaluation_function, job_queue, res_queue):
    worker_id = mp.current_process()._identity[0]
    while True:
        args = job_queue.get()
        res = evaluation_function(tasks=args["tasks"], candidates=args["candidates"], worker_id=worker_id)
        res_queue.put(res)


# %% [markdown]
# ### for GAME

# %%
def sample_n_random_BTs(rng, n, config):
    initial_blues = []
    keys = set()
    while len(keys) < n:
        bt, _ = random_sampling(rng, config)
        key = bt["bt"].to_txt()
        if key not in keys:
            keys.add(key)
            initial_blues.append(bt)
    return initial_blues

def create_tasks(bts, generation):
    tasks = []
    for i, bt in enumerate(bts):
        bt["generation"] = generation
        tasks.append(bt)
    return tasks 


# %% [markdown]
# ### Make Videos

# %%
def organize_points_on_grid(points, x_size, y_size):
    """
    Organizes 2D points into a square grid.

    Args:
        points: List of 2D points (e.g., [(x1, y1), (x2, y2), ...])
        grid_size: Number of grid cells in each dimension.

    Returns:
        A list of 2D points organized on the grid.
    """

    # 1. Create a grid of coordinates
    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, x_size), np.linspace(0, 1, y_size))
    grid_points = np.column_stack((x_grid.flatten(), y_grid.flatten())) 

    # 2. Initialize a KDTree for efficient nearest neighbor search
    tree = cKDTree(grid_points)
    # 3. Assign points to grid cells
    organized_points = []
    for point in points:
        # Find the nearest empty grid cell
        distances, indices = tree.query(point, k=int(x_size*y_size))
        i = 0
        while i < len(indices) and tuple(grid_points[indices[i]]) in organized_points: 
            i += 1
        organized_points.append(tuple(grid_points[indices[i]]))

    return organized_points


# %%
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB (OpenCV uses BGR, Pillow uses RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert frame to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        frames.append(pil_image)
    return frames


# %% [markdown]
# #### in training

# %%
def make_videos(video_eval_fn, name, me, n_steps):  # no to date 
    video_folder = me.save_folder + "videos_" + name
    utils.create_folder(video_folder)
    for i, bt in tqdm(me.archive.archives[0].cells_solution.items()):
        if not os.path.exists(video_folder + f"/{i}_0_0.mp4"):
            
            video_eval_bts_fn([bt.to_txt()], True, video_folder + f"/{i}")
    B = me.archive.archives[0].cells_behavior[:len(me.archive.archives[0].cells_solution)]
    if B.shape[1] > 2:
        reducer = umap.UMAP(n_components=2, random_state=42)  # 2D reduction
        embedding = reducer.fit_transform(B)
        E = embedding
    else:
        E = B 
    E = (E - np.min(E, axis=0)) / (np.max(E, axis=0) - np.min(E, axis=0))
    points = E
    x, y = 16, 9
    n_columns = x 
    n_rows = y 

    organized_points = organize_points_on_grid(points, x, y)
    dis_pos = np.array(np.array(organized_points)*np.array([x-1, y-1]), dtype=int)
    W, H = np.max(dis_pos, axis=0) - np.min(dis_pos, axis=0) + 1
    image_size = 100
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
    
    for i in tqdm(range(len(B))):
        frames = extract_frames(video_folder + f"/{i}_0_0.mp4")
        for t in range(n_steps//timestep):
            (x_pos, y_pos) = dis_pos[i]
            y = y_pos * (image_size+2*pad) 
            x = x_pos * (image_size+2*pad) 
            img = frames[t]
            img = img.resize((image_size, image_size))
            images[t].paste(Image.new("RGBA", (image_size+2*pad, image_size+2*pad), "gray"), (x, y))
            images[t].paste(img, (x+pad, y+pad))
    
    path = video_folder + f'/umap'
    Tmax = 10
    images[0].save( path + '.gif', save_all=True, append_images=images[1:], optimize=True, duration=Tmax/24, loop=0)

    videodims = images[0].size
    fourcc = cv2.VideoWriter_fourcc(*'avc1')    
    video = cv2.VideoWriter(path + ".mp4", fourcc, 18, videodims)
    img = Image.new('RGB', videodims, color = 'darkred')
    for image in images:
        video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    video.release()
    # make a nice html page to 
    videos = []
    for i in range(len(B)):
        c, r = dis_pos[i, 1], dis_pos[i, 0]
        description = f"fitness: {100*me.archive.archives[0].cells_fitness[i][0]:1.0f}%"
        if B.shape[1] == 2:
            remaining_health, duration = B[i]
        else:
            idx = me.archive.archives[0].cells_log_id[i]
            remaining_health, duration = me.log[idx//me.batch_size][idx%me.batch_size]["other"]
        description += "\n" + "duration: " + f"{100 * duration:1.0f}%" 
        description += "\n" + "remaining_health: " + f"{100 * remaining_health: 1.0f}%"  
        description += "\n" + me.archive.archives[0].cells_solution[i].to_pretty_txt()
        video = {
            "path": video_folder + f"/{i}_0_0.mp4",
            "title": str(i), 
            "description": description, 
            "c_start": c+1,
            "c_end": c+2,
            "r_start": r+1,
            "r_end": r+2,
        }
        videos.append(video)
    n_videos = len(videos)
    html = get_html(n_videos)
    css = get_css(n_columns, n_rows)
    js = get_js(videos)
    with open(video_folder + "/main.html", 'w') as file:
        file.write(html)
    with open(video_folder + "/styles.css", 'w') as file:
        file.write(css)
    with open(video_folder + "/script.js", 'w') as file:
        file.write(js)


# %% [markdown]
# #### for tournament

# %%
def make_videos_for_tournament(video_eval_fn, video_folder, tournament, mini_tournament_size, n_steps, n_groups, group_size, fixed_starting_sector, fixed_unit_types, image_size=100):
    utils.create_folder(video_folder)

    for (i,j), duel in tqdm(tournament.items()):
        if not os.path.exists(video_folder + f"/{i}_{j}_0.mp4"):
            a_bt, e_bt, sectors, unit_types = get_jax_params(duel["red"], duel["blue"], n_groups, group_size, fixed_starting_sector, fixed_unit_types)
            unit_starting_sectors = jnp.array([sectors])
            unit_types = jnp.array([unit_types], dtype=jnp.uint8)
            video_eval_fn([a_bt], [e_bt], unit_starting_sectors, unit_types, True, video_folder + f"/{i}_{j}")
    
    W, H = mini_tournament_size, mini_tournament_size
    timestep = 1
    pad = 2
    W *= (image_size+2*pad)
    H *= (image_size+2*pad)
    if True: # not os.path.exists(video_folder + f'/umap.mp4'):
        images = []
        for t in range(n_steps//timestep):
            new = Image.new("RGBA", (W, H))
            draw = ImageDraw.Draw(new)
            new.paste(Image.new("RGBA", new.size, "WHITE"), (0,0))
            images.append(new)
        
        for (i, j) in tqdm(tournament.keys()):
            frames = extract_frames(video_folder + f"/{i}_{j}_0.mp4")
            for t in range(n_steps//timestep):
                (x_pos, y_pos) = j, mini_tournament_size-i-1
                y = y_pos * (image_size+2*pad) 
                x = x_pos * (image_size+2*pad) 
                img = frames[t]
                img = img.resize((image_size, image_size))
                images[t].paste(Image.new("RGBA", (image_size+2*pad, image_size+2*pad), "gray"), (x, y))
                images[t].paste(img, (x+pad, y+pad))
        
        path = video_folder + f'/umap'
        Tmax = 10
        images[0].save( path + '.gif', save_all=True, append_images=images[1:], optimize=True, duration=50, loop=0)
    
        videodims = images[0].size
        fourcc = cv2.VideoWriter_fourcc(*'avc1')    
        video = cv2.VideoWriter(path + ".mp4", fourcc, 20, videodims)
        img = Image.new('RGB', videodims, color = 'darkred')
        for image in images:
            video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        video.release()
    # make a nice html page to 
    videos = []
    for (i, j), duel in tqdm(tournament.items()):
        c, r = mini_tournament_size-i-1, j
        Ldescription = f"red fitness: {100*duel['red_fitness']:1.0f}%"
        Rdescription = f"blue fitness: {100*(duel['blue_fitness']):1.0f}%"
        Ldescription += "\n" + duel["red"]["bt"].to_pretty_txt()
        Rdescription += "\n" + duel["blue"]["bt"].to_pretty_txt()
        video = {
            "path": video_folder + f"/{i}_{j}_0.mp4",
            "title": f"{i} vs {j}", 
            "rightDescription": Rdescription,
            "leftDescription": Ldescription,
            "c_start": c+1,
            "c_end": c+2,
            "r_start": r+1,
            "r_end": r+2,
        }
        videos.append(video)
    n_videos = len(videos)
    html = get_bi_html(n_videos)
    css = get_bi_css(mini_tournament_size, mini_tournament_size)
    js = get_bi_js(videos)
    with open(video_folder + "/main.html", 'w') as file:
        file.write(html)
    with open(video_folder + "/styles.css", 'w') as file:
        file.write(css)
    with open(video_folder + "/script.js", 'w') as file:
        file.write(js)


# %% [markdown]
# ## BT functions

# %%
def cosine_distance(x, Y):
    """
    2 * cosine_distance(x,y) = ||x-y||² if ||x|| = ||y|| = 1
    """
    dot_product = np.dot(Y, x)
    norm_x = np.linalg.norm(x)
    norm_Y = np.linalg.norm(Y, axis=1)
    return 1-dot_product/(norm_x * norm_Y)

def compare_fitness_bt(new, old):
    return (old is None) or (new[0] > old[0]) or (new[0] == old[0] and new[1] > old[1])

def compare_fitness_bt_nobias(new, old):
    return (old is None) or (new[0] > old[0])

def compare_fitness_no_quality(new, old):
    return (old is None)


# %%
def sample_random_bt(rng, max_leaves, max_depth, max_children):
    child = BT.random(rng, max_depth, max_children)
    if child.n_leaves() > max_leaves:
        return sample_random_bt(rng, max_leaves, max_depth, max_children)
    else:
        return child, {"type": "random"}

def random_sampling(rng, config):
    """
    In Adversarial Coevolutionary Illumination with Generational Adversarial MAP-Elites, 
    only the bt is evaluated; the unit_starting_sectors and unit_types pass through the 
    variation operators but are not applied, and are replaced by fixed values. They are 
    remnants of historical experiments that were kept intact to avoid incorporating bugs.
    """
    bt, _ = sample_random_bt(rng, **config["bt"])
    n_groups = config["n_groups"]
    unit_starting_sectors = rng.random((n_groups, 4))
    unit_types = rng.integers(0, len(grammar.unit_types), (n_groups,))
    return {'bt': bt, "unit_starting_sectors": unit_starting_sectors, "unit_types": list(unit_types)}, {"type": "random"}
    

def bt_variation(rng, p1, p2, crossover_probability, mutation_probas, max_leaves):
    if rng.random() < crossover_probability:  # crossover between two random elites
        child = p1.crossover(rng, p2)
        kind = "crossover"
    else:  # mutation of a random elite
        child, mutation_type = p1.mutation(rng, mutation_probas)
        kind = mutation_type
    if child.n_leaves() > max_leaves:
        return bt_variation(rng, p1, p2, crossover_probability, mutation_probas, max_leaves)
    else:
        return child, {"type": kind}

def iso_line_dd(rng, x, y, iso_sigma, line_sigma, dims):
    candidate = x + rng.normal(np.zeros(dims), iso_sigma) + rng.normal(np.zeros(dims), line_sigma) * (y-x) 
    return np.clip(candidate, 0, 1)
    
def crossover(rng, p1, p2, config):
    bt, kind = bt_variation(rng, p1["bt"], p2["bt"], **config["bt"])
    n_groups = config["n_groups"]
    unit_starting_sectors = iso_line_dd(rng, p1["unit_starting_sectors"], p2["unit_starting_sectors"], config['iso_sigma'], config['line_sigma'], (n_groups, 4))
    unit_types = []
    for i in range(n_groups):
        unit_type = p1["unit_types"][i] if rng.random() < config["unit_type_crossover"] else p2["unit_types"][i]
        if rng.random() < config["unit_type_mutation"]:
            unit_type = rng.integers(0, 5)
        unit_types.append(unit_type)
    return {'bt': bt, "unit_starting_sectors": unit_starting_sectors, "unit_types": unit_types}, {"type": {"bt": kind, "others": "variation"}}


# %%
def get_jax_params(reds, blues, n_groups, group_size, fixed_starting_sector=None, fixed_unit_types=None):
    sectors = []
    for blue_sector in blues["unit_starting_sectors"]:
        if fixed_starting_sector is None:
            sector = np.array([0., 0., 0., 0.1]) + blue_sector*np.array([0.7, 0.4, 0., 0.])
        else:
            sector = np.array(fixed_starting_sector) 
        sectors += [sector] * group_size
    for red_sector in reds["unit_starting_sectors"]:
        if fixed_starting_sector is None:
            red_sector = np.array([0., 0., 0., 0.1]) + red_sector*np.array([0.7, 0.4, 0., 0.])
        else:
            red_sector = np.array(fixed_starting_sector) 
        red_sector[1] = 1 - red_sector[1] - red_sector[3]  # vertical symmetry
        sectors += [red_sector] * group_size
    if fixed_unit_types is None:
        unit_types = np.concatenate([[unit_type] * group_size for unit_type in blues["unit_types"] + reds["unit_types"]]) 
    else:
        unit_types = np.concatenate([[unit_type] * group_size for unit_type in fixed_unit_types + fixed_unit_types]) 
    return blues["bt"].to_txt(), reds["bt"].to_txt(), sectors, unit_types 

def evaluate_bi_bt_batch(eval_bt_fn, use_embedding, n_groups, group_size, fixed_starting_sector, fixed_unit_types, tasks, candidates, worker_id=None):  
    assert len(tasks) == len(candidates)
    a_bts, e_bts, unit_starting_sectors, unit_types= [], [], [], []
    for i in range(len(tasks)):
        if tasks[i]["config"]["generation"] == "red":
            blues, reds = tasks[i]["config"], candidates[i]["value"]
        else:
            blues, reds = candidates[i]["value"], tasks[i]["config"]
        
        a_bt, e_bt, sectors, unit_type = get_jax_params(reds, blues, n_groups, group_size, fixed_starting_sector, fixed_unit_types)
        a_bts.append(a_bt)
        e_bts.append(e_bt)
        unit_starting_sectors.append(sectors)
        unit_types.append(unit_type) 
        
    unit_starting_sectors = jnp.array(unit_starting_sectors)
    unit_types = jnp.array(unit_types, dtype=jnp.uint8)
    measures = eval_bt_fn(a_bts, e_bts, unit_starting_sectors, unit_types)  # put blue as allies and red as enemies
    evaluations = []
    for i, (task, candidate) in enumerate(zip(tasks, candidates)):
        if task["config"]["generation"] == "red":
            f, b1 = float(1-measures.ally_health[i]), float(measures.enemy_health[i])
        else:
            f, b1 = float(1-measures.enemy_health[i]), float(measures.ally_health[i]) 
            
        if use_embedding:
            b = np.array(measures.embedding[i].reshape(-1))
            other = np.array([b1, float(measures.duration[i])])
        else:
            b = np.array([b1, float(measures.duration[i])])
            other = np.array(measures.embedding[i].reshape(-1))
        duration = int(measures.duration[i] * len(measures.actions_id))
        actions_distributions = {'red': np.unique(measures.actions_id[:duration, i, n_agents//2:], return_counts=True), "blue": np.unique(measures.actions_id[:duration, i, :n_agents//2], return_counts=True)}
        evaluation = {"task_id": task["id"], "id": None, "solution": candidate["value"],
                      "fitness": [f, -candidate["value"]["bt"].size], 
                      "behavior": b, "actions_id": actions_distributions, "other": other}
        evaluations.append(evaluation)    
    return evaluations


# %% [markdown]
# ## Tournament

# %%
def compute_tournament(config, reds, blues):
    tournament = {}
    for i, red_team in enumerate(reds):
        for j, blue_team in enumerate(blues):
            blue_team["generation"] = "red"
            tournament[(i,j)] = {"candidate": {"value":red_team}, "task": {"id": j, "config": blue_team}} 
    batch_size = 10
    for i in tqdm(range(0, len(tournament), batch_size)):
        keys = list(tournament.keys())[i:i+batch_size]
        if len(keys) < batch_size:
            keys += [keys[-1]]*(batch_size-len(keys))
        candidates = [tournament[key]["candidate"] for key in keys]
        tasks = [tournament[key]["task"] for key in keys]
        evaluations = evaluate_bi_bt_batch(**config["evaluation_config"], tasks=tasks, candidates=candidates)
        for j, key in enumerate(keys):
            tournament[key]["eval"] = evaluations[j]
    return tournament


# %%
def compute_pareto_elites(score, strict=True):
    elites = []
    for i in range(len(score)):
        dominated = False
        for j in range(len(score)):
            if i != j:
                if np.all(score[j] >= score[i]) and np.any(score[j] > score[i]):
                    dominated = True
                    break
        if not dominated:
            new = True
            if strict:
                for elite in elites:
                    if np.all(score[elite] == score[i]):
                        new = False
                        break
            if new:
                elites.append(i)  
    return elites

def compute_experts(score, strict=True):
    elites = []
    expertise = np.array([np.sum([score[j] > score[i] for j in range(len(score))], axis=0) == 0 for i in range(len(score))])
    for i in range(len(score)):
        expert = True
        for j in range(len(score)):
            if np.sum(expertise[j].dot(expertise[i])) == np.sum(expertise[i]) and np.sum(expertise[j]) > np.sum(expertise[i]):
                expert = False
                break 
        if strict: 
            for j in elites:
                if np.all(expertise[j] == expertise[i]):
                    expert = False
                    break
        if expert:
            elites.append(i)  
    return elites


# %%
def make_tournament_plot(n_red, n_def, tournament, gen_id, path, sorting=True):
    F = np.zeros((n_red, n_def))
    for (i,j), val in tournament.items():
        red_f = val["eval"]["fitness"][0]
        blue_f = 1-val["eval"]["other"][0]
        F[i,j] = 100*red_f if red_f > blue_f else -100*blue_f
    if sorting:
        red_indices = jnp.argsort(jnp.mean(F, axis=1))
        blue_indices = jnp.argsort(-jnp.mean(F, axis=0))
    plt.subplots(figsize=(10,9))
    plt.pcolor(F[red_indices][:, blue_indices] if sorting else F, vmin=-100, vmax=100, cmap="coolwarm")
    plt.xticks([])
    plt.yticks([])
    plt.ylabel("reds")
    plt.xlabel("blues")
    plt.colorbar(label="Loosing Side Depleted Health (%)")
    if path is not None:
        utils.savefig(f"tournament_{gen_id}", path, timestamp=False)
    plt.close()
    
    red_dominant = compute_pareto_elites(F, True)
    blue_dominant = compute_pareto_elites(-np.transpose(F), True)
    red_experts = compute_experts(F)
    blue_experts = compute_experts(-np.transpose(F))
    print("Non-dominated reds/blues ", len(red_dominant), "/", len(blue_dominant))
    print("Experts reds/blues ", len(red_experts), "/", len(blue_experts))
    if path is not None:
        utils.save_pickle(path+f"/experts_counts.pk", {"ND-red": len(red_dominant),"ND-def": len(blue_dominant), "red-experts": len(red_experts), "def-experts": len(blue_experts)})
    return F

def mini_tournament_videos(path, mini_tournament_size, gen_id, tournament, F, n_groups, group_size, fixed_starting_sector, fixed_unit_types, sorting=True):
    if sorting:
        red_indices = jnp.argsort(jnp.mean(F, axis=1))
        blue_indices = jnp.argsort(-jnp.mean(F, axis=0))
    else:
        red_indices = jnp.arange(F.shape[0])
        blue_indices = jnp.arange(F.shape[1])
    indices = np.linspace(0, min(len(red_indices), len(blue_indices))-1, mini_tournament_size, dtype=np.int32)
    sub_atackers = [reds[i] for i in red_indices[indices]]
    sub_blues = [blues[i] for i in blue_indices[indices]]
    video_tournament = {}
    for i, red_bt in enumerate(sub_atackers):
        for j, blue_bt in enumerate(sub_blues):
            video_tournament[(i,j)] = {"red": red_bt, "blue": blue_bt, 
                                       "red_fitness": tournament[(int(red_indices[indices][i]), int(blue_indices[indices][j]))]["eval"]["fitness"][0],
                                       "blue_fitness": 1-tournament[(int(red_indices[indices][i]), int(blue_indices[indices][j]))]["eval"]["other"][0],
                                      }
    video_folder = path + f"/tournament_{gen_id}"
    make_videos_for_tournament(video_eval_bi_bts_fn, video_folder, video_tournament, mini_tournament_size, n_steps=100, n_groups=n_groups, group_size=group_size, fixed_starting_sector=fixed_starting_sector, fixed_unit_types=fixed_unit_types, image_size=1000//mini_tournament_size)
    plt.subplots(figsize=(10,9))
    plt.pcolor(F[red_indices[indices]][:, blue_indices[indices]], vmax=100, cmap="coolwarm")
    plt.xticks([])
    plt.yticks([])
    plt.ylabel("reds")
    plt.xlabel("blues")
    plt.colorbar(label="blues Depleted Health (%)")
    utils.savefig(f"mini_tournament_{gen_id}", path + f"/tournament_{gen_id}/",  timestamp=False)
    plt.close()


# %% [markdown]
# # Config 

# %%
n_agents = 10 * 32
n_groups = 5
n_bts = 10  # // evaluations in jax
group_size = n_agents//(2*n_groups)

parabellum_config = {
    "n_bts": n_bts,
    "places": ["blank"]*n_bts,
    "bt_max_size": 100,
    "map_size": 100,
    "n_agents": n_agents,
    "n_allies": n_agents//2,
    "n_steps": 100,
    "frames_indexes": jnp.array([19, 39, 59, 79, 99]),
    "units_resolution_scale": 1,
    "seed": 0,
    "max_map_targets": len(grammar.targets)*2,  # currently used for the self-set targets 
}

eval_bi_bts_fn = eval_bi_bts_factory(parabellum_config)
evaluate_bi_bt_batch = partial(evaluate_bi_bt_batch, eval_bi_bts_fn)
video_parabellum_config = deepcopy(parabellum_config)
video_parabellum_config["n_bts"] = 1
video_parabellum_config["places"] = ["blank"]
video_eval_bi_bts_fn = eval_bi_bts_factory(video_parabellum_config)


# %%
def get_config(parabellum_config, budget, tasks, n_cells, n_groups, use_embedding=True, use_diversity_only=False, use_quality_only=False, seed_id=0, bias_for_small_BT=True):
    config = {  # BT
        "tasks": tasks,
        "budget": budget,
        'batch_size': parabellum_config["n_bts"],
        'init_elites': 100,
        'seed': utils.seeds[seed_id],
        'verbose': True,
        'log_interval': 10_000,
        "intermediate_videos": False,
        "first_video": False,
        "last_video": False,
        "video_eval_fn": video_eval_bi_bts_fn,
        "avoid_repetition": False,
        "parallel": False,
        "n_proc": N_PROC,
        "archive_config": {
            "use_growing": not use_quality_only,
            'n_cells': n_cells,
            'n_behavior_dim': len(parabellum_config["frames_indexes"])*512 if use_embedding else 2,
            "n_solution_dim": None,
            "use_redristribution": False,
            "use_collection": False,
            "use_repair": True,
            "compare_fitness": (compare_fitness_bt if bias_for_small_BT else compare_fitness_bt_nobias) if not use_diversity_only else compare_fitness_no_quality,
            "distance_function": cosine_distance,
        },
    
        "sample_random_function": random_sampling,
        "random_sampling_config": {"bt": {"max_leaves": parabellum_config["bt_max_size"], 'max_depth': 3,  'max_children': 3}, "n_groups": n_groups},
    
        "crossover_and_mutation_function": crossover,
        "crossover_and_mutation_config": {"bt": {"crossover_probability": 0.3, 
                                                "mutation_probas": {'weak mutation': 0.1, 'strong mutation': 0.1, 'addition': 0.3, 'deletion':0.5},
                                                "max_leaves": parabellum_config["bt_max_size"]},
                                          "n_groups": n_groups, "iso_sigma": 0.01, "line_sigma": 0.2, "unit_type_mutation": 0.05, "unit_type_crossover": 0.5,
                                         },
        
        "evaluation_function": evaluate_bi_bt_batch, 
        "evaluation_config" : {"use_embedding": use_embedding, "n_groups": n_groups, "group_size": parabellum_config["n_allies"]//n_groups, 
                               "fixed_starting_sector": [0.33, 0.1, 0.33, 0.2], "fixed_unit_types": [0, 1, 2, 3, 4]},
    }
    return config 


# %% [markdown]
# # Run

# %%
n_tasks = 100  #100
n_gen = 0 # 20
budget = 100_000  # 100_000
seed_id = 0
n_cells = 25  # 25

use_boostrap = True
bias_for_small_BT = True  # True for GAME-MO, False for GAME-SO
use_embedding = True  # False for GAME (no VEM) 
use_diversity_only = False  
use_quality_only = False 

assert not use_diversity_only or not use_quality_only
mini_tournament_size = 5
do_mini_tournament = False  # only do mini tournament if doing bootstrap

# %%
mini_config = {"n_tasks": n_tasks, "n_gen": n_gen, "budget": budget, "seed_id": seed_id, "n_cells": n_cells, 
               "use_boostrap": use_boostrap, "bias_for_small_BT": bias_for_small_BT, "use_embedding": use_embedding, 
               "use_diversity_only": use_diversity_only, 'use_quality_only': use_quality_only,
              }

# %% [markdown]
# ## GAME

# %%
if __name__ == "__main__":
    # config 
    config = get_config(parabellum_config, budget, None, n_cells, n_groups, 
                        use_embedding=use_embedding, use_diversity_only=use_diversity_only, use_quality_only=use_quality_only, 
                        seed_id=seed_id, bias_for_small_BT=bias_for_small_BT)
    main_folder = utils.create_save_folder()
    utils.save_pickle(main_folder + f"/mini_config.pk", mini_config)
    rng = np.random.default_rng(config["seed"])
    # init
    current_bts = sample_n_random_BTs(rng, n_tasks, config['random_sampling_config'])
    bootstrap_evaluations = []
    # main loop
    for gen_id in range(n_gen):
        print(f"Gen {gen_id}: {len(set([bt['bt'].to_txt() for bt in current_bts]))} different BTs.") 
        config["save_folder"] = main_folder + f"/gen_{gen_id}/"
        utils.create_folder(config["save_folder"])
        config["tasks"] = create_tasks(current_bts, generation=["red", "blue"][gen_id%2])  # gen == atk means optimizing the reds and fixing the blues
        me = MT_GAME(config)
        if use_boostrap:
          me.update_archive(bootstrap_evaluations)
        # run gen
        me.run()
        # compute next gen 
        elites = [{"fitness": None, "bt": None} for _ in range(n_tasks)]
        if use_quality_only:
            for task_id, archive in enumerate(me.archive.archives):
                for cell_id in range(config["archive_config"]['n_cells']):
                    if cell_id in archive.non_empty_cells:
                        elites[task_id]["fitness"] = archive.cells_fitness[cell_id]
                        elites[task_id]["bt"] = archive.cells_solution[cell_id]
        else:  # Quality and Diversity
            behaviors = jnp.concatenate([archive.cells_behavior[archive.non_empty_cells] for archive in me.archive.archives])
            kmeans = KMeans(n_clusters=n_tasks).fit(preprocessing.normalize(behaviors))   # for normalized vectors euclidian distance is equivalent to cosine distance ||x-y||² = 2 (1 - x.y)
            centroids = kmeans.cluster_centers_
            tree = cKDTree(centroids)
            elites_bt_txt = [""] * n_tasks
            distances_to_centroids = np.ones(n_tasks) * np.inf
            for task_id, archive in enumerate(me.archive.archives):
                for cell_id in range(config["archive_config"]['n_cells']):
                    if cell_id in archive.non_empty_cells:
                        distance, c_id = tree.query(archive.cells_behavior[cell_id]/jnp.linalg.norm(archive.cells_behavior[cell_id]), k=1)
                        if use_diversity_only:
                            if distance < distances_to_centroids[c_id] and archive.cells_solution[cell_id]["bt"].to_txt() not in elites_bt_txt:
                                distances_to_centroids[c_id] = distance 
                                elites[c_id]["fitness"] = archive.cells_fitness[cell_id]
                                elites[c_id]["bt"] = archive.cells_solution[cell_id]
                                elites_bt_txt[c_id] = archive.cells_solution[cell_id]["bt"].to_txt()
                        else:  # diversity and quality 
                            if compare_fitness_bt(archive.cells_fitness[cell_id], elites[c_id]["fitness"]) and archive.cells_solution[cell_id]["bt"].to_txt() not in elites_bt_txt:
                                elites[c_id]["fitness"] = archive.cells_fitness[cell_id]
                                elites[c_id]["bt"] = archive.cells_solution[cell_id]
                                elites_bt_txt[c_id] = archive.cells_solution[cell_id]["bt"].to_txt()
        current_bts = [elite["bt"] for elite in elites if elite["bt"] is not None]
        # compute tournament
        if gen_id % 2 == 1:  # blues 
            blues = current_bts
            reds = [task for task in me.tasks] 
        else:
            reds = current_bts
            blues = [task for task in me.tasks] 
        utils.save_pickle(config["save_folder"] + f"elites_{gen_id}.pk", {"reds": reds, "blues": blues})
        
        # bootstrap next generation
        if use_boostrap:
            tournament = compute_tournament(config, reds, blues)
            utils.save_pickle(config["save_folder"] + f"tournament_{gen_id}.pk", tournament)
            bootstrap_evaluations = []
            for (i,j), val in tournament.items():
                if gen_id % 2 == 1:  # blues 
                    solution = reds[i]
                    f = [val["eval"]["fitness"][0], -solution["bt"].size]  # blue depleted health 
                    task_id = j
                else:
                    solution = blues[j]
                    f = [1-val["eval"]["other"][0], -solution["bt"].size]  # red depleted health
                    task_id = i 
                bootstrap_evaluations.append({"id": -1, "task_id": task_id, "fitness": f, "behavior": val["eval"]["behavior"], "solution": solution})
        
            F = make_tournament_plot(len(reds), len(blues), tournament, gen_id, config["save_folder"])
            if do_mini_tournament:
                mini_tournament_videos(config["save_folder"], mini_tournament_size, gen_id, tournament, F, n_groups, group_size, config["evaluation_config"]["fixed_starting_sector"], config["evaluation_config"]["fixed_unit_types"])

# %% [markdown]
# ## Intergenerational tournament

# %%
if __name__ == "__main__":
    Elites = [utils.load_pickle(main_folder + f"/gen_{gen_id}/elites_{gen_id}.pk") for gen_id in range(n_gen)]
    
    blues, reds = [], []
    for gen_id in [i for i in range(0, n_gen, 2)]:
        for blue in Elites[gen_id]["blues"]:
            blues.append(blue)
        for red in Elites[gen_id]["reds"]:
            reds.append(red)
    
    config = get_config(parabellum_config, 0, None, n_cells=None, n_groups=n_groups, use_embedding=True)
    tournament = compute_tournament(config, reds, blues)
    utils.save_pickle(main_folder + f"generational_tournament.pk", tournament)

# %%
