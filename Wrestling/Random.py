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
from jax import vmap, jit
import subprocess
import cv2
from PIL import Image, ImageDraw 
import umap.umap_ as umap  # pip install umap-learn
from scipy.spatial import cKDTree
from sklearn import preprocessing
import misc_plot
from scipy.signal import convolve2d
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

# %%
import sys
sys.path.append("wrestlers")  
sys.path.append("wrestlers/world_data")  
from wrestlers.sample_robot import sample_robot
import envs
from clip import CLIP, get_image

# %%
N_PROC = 32

# %%
import contextlib
import io

@contextlib.contextmanager
def suppress_output():
    # Open null device
    null_fd = os.open(os.devnull, os.O_RDWR)
    # Save original stdout and stderr file descriptors
    save_stdout = os.dup(1)
    save_stderr = os.dup(2)
    # Redirect stdout and stderr to null device
    os.dup2(null_fd, 1)
    os.dup2(null_fd, 2)
    try:
        yield
    finally:
        # Restore original stdout and stderr
        os.dup2(save_stdout, 1)
        os.dup2(save_stderr, 2)
        # Close file descriptors
        os.close(null_fd)
        os.close(save_stdout)
        os.close(save_stderr)


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
        return self.cells_solution[p], self.cells_log_id[p]

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
        evaluation["is_elite"] = True
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
            changed = True
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
        became_elites = []
        for evaluation in evaluations:
            if evaluation is not None:
                self.archives[evaluation['task_id']].add_evaluation(evaluation)
                if evaluation['task_id'] not in self.non_empty_archive:
                    self.non_empty_archive.append(evaluation['task_id'])
        
    def sample_parents(self):
        p1_a_id, p2_a_id = self.rng.choice(self.non_empty_archive, 2)
        p1, p1_id = self.archives[p1_a_id].sample_parent()
        p2, p2_id = self.archives[p2_a_id].sample_parent()
        return p1, p2, (p1_a_id, p1_id), (p2_a_id, p2_id)

    def n_elites(self):
        return np.sum([archive.n_elites() for archive in self.archives])


# %% [markdown]
# ## MT GAME

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
        self.use_logging = config["use_logging"]
        self.init_elites = config["init_elites"]
        self.avoid_repetition = config["avoid_repetition"] and (config["archive_config"]["n_solution_dim"] is None)  # only for BTs which are discrete
        self.alpha = config["alpha"] if "alpha" in config else None  # scale of the behavior space for the benchmark comparison 

        self.intermediate_videos = config["intermediate_videos"]
        self.first_video = config["first_video"]
        self.last_video = config["last_video"]
        self.make_videos_function = None #partial(make_videos, config["video_eval_fn"])
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
        archive_save["tasks"] = self.tasks
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
            if ev is not None:
                ev["id"] = self.it
                self.it += 1
        self.archive.update(evaluations)
        if self.use_logging:
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

        last_it = self.it
        while self.it < self.budget:
            if self.parallel:  # collect the evaluation
                evaluated_candidates = res_queue.get()
            else:
                tasks, candidates = job_queue.pop(0)
                evaluated_candidates = self.evaluation_function(tasks=tasks, candidates=candidates)
                
            has_changed = self.update_archive(evaluated_candidates)
                                    
            if self.it % self.log_interval == 0:  # save archive 
                #utils.save_pickle(self.save_folder + "/xp.pk", self)
                self.save_archive()
                if self.archive.n_elites() == self.archive.archives[0].n_cells and self.intermediate_videos:
                    make_videos(str(it*self.batch_size), self, parabellum_config["n_steps"])
            
            if self.verbose:  # update loading bar info
                loading_bar.update(self.it - last_it)
                last_it = self.it
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
# ## Wrestler ENV

# %% [markdown]
# ### Fitness and behaviors

# %%
def cosine_distance(x, Y):
    """
    2 * cosine_distance(x,y) = ||x-y||² if ||x|| = ||y|| = 1
    """
    dot_product = np.dot(Y, x)
    norm_x = np.linalg.norm(x)
    norm_Y = np.linalg.norm(Y, axis=1)
    return 1-dot_product/(norm_x * norm_Y)

def compare_fitness(new, old):
    return (old is None) or (new > old)

def compare_fitness_no_quality(new, old):
    return (old is None)


# %% [markdown]
# ### Sampling

# %%
def add_phase(rng, body):
    return body + np.where(body > 2, rng.integers(0, 2, size=body.shape)*2, 0.)

def sample_random_wrestler(rng, config):
    width = config["width"]
    wrestler = add_phase(rng, sample_robot(rng, (width, width)))
    return {"wrestler": wrestler}, "random"


# %%
def get_wrestler_key(wrestler):
    return tuple(wrestler['wrestler'].reshape(-1))
    
def sample_n_random_wrestlers(rng, n, config):
    width = config["width"]
    initial_blues = []
    keys = set()
    while len(keys) < n:
        wrestler = {"wrestler": add_phase(rng, sample_robot(rng, (width, width)))}
        key = get_wrestler_key(wrestler)
        if key not in keys:
            keys.add(key)
            initial_blues.append(wrestler)
    return initial_blues

def create_tasks(solutions, generation):
    tasks = []
    for i, s in enumerate(solutions):
        s["generation"] = generation
        tasks.append(s)
    return tasks 


# %% [markdown]
# ### Variation Operators

# %%
def is_connected(cells, excluded_cell=None):
    cells = [c for c in cells if c != excluded_cell]
    if len(cells) == 0:
        return True
    visited = set()
    queue = [cells[0]]
    visited.add(tuple(cells[0]))
    while queue:
        i,j = queue.pop(0)
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = i + dr, j + dc
            if (nr, nc) in cells and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc)) 
    return len(visited) == len(cells)

def compute_removable_cells(wrestler):
    if np.sum(wrestler>0) < 2:
        return []
    cells = [(i,j) for i, j in zip(*np.where(wrestler>0)) if (i,j)]
    removable_cells = []
    for cell in cells:
        if is_connected(cells, excluded_cell=cell):
            removable_cells.append(cell)
    return removable_cells

def wrestler_deletion(rng, wrestler):
    removable_cells = compute_removable_cells(wrestler)
    i, j = rng.choice(removable_cells)
    while np.sum(wrestler>2) == 1 and wrestler[i,j] > 2:  # if we selected the only actuators (we know there are more than one cells)
        i, j = rng.choice(removable_cells)
    wrestler[i,j] = 0.
    return wrestler

def wrestler_addition(rng, wrestler):
    kernel = np.array([[0,1,0], [1,0,1], [0,1,0]])
    available_cells = [(i,j) for i, j in zip(*np.where((wrestler ==0) & (convolve2d(wrestler>0, kernel, mode="same", boundary="fill")>0)))]
    i, j = rng.choice(available_cells)
    wrestler[i,j] = rng.integers(1, 7)
    return wrestler

def wrestler_mutation(rng, wrestler):
    valid_cells = [(i,j) for i, j in zip(*np.where((wrestler>0)))]
    i, j = rng.choice(valid_cells)
    if np.sum(wrestler>2) == 1 and wrestler[i,j] > 2:  # if we selected the only actuators 
        choices = [3, 4, 5, 6]
    else:
        choices = [1, 2, 3, 4, 5, 6]
    choices.remove(wrestler[i,j])
    wrestler[i,j] = rng.choice(choices)
    return wrestler

def mutate_wrestler(rng, wrestler, _, config):  # second parent ignored at the moment (i.e., no crossover)
    mutation_probas = config["mutation_probas"]
    new_wrestler = np.copy(wrestler["wrestler"])
    mutations_type = []
    for _ in range(config['n_mutations']):
        r = rng.random()
        if (r < mutation_probas['deletion']) and np.sum(new_wrestler>2) > 1:  # deletion
            new_wrestler = wrestler_deletion(rng, new_wrestler)
            mutations_type.append("deletion")
        elif (r < mutation_probas['deletion'] + mutation_probas['addition']) and np.sum(new_wrestler==0)>0:  # addition
            new_wrestler = wrestler_addition(rng, new_wrestler)
            mutations_type.append("addition")
        else:  # mutation
            new_wrestler = wrestler_mutation(rng, new_wrestler)
            mutations_type.append("mutation")
    return {"wrestler": new_wrestler}, {"type": mutations_type}


# %% [markdown]
# ### Evaluation

# %%
def make_env(env_id, blue_wrestler, red_wrestler, seed=None, render_mode="rgb_array", resolution=(224, 224)):
    """
    Creates a callable function that will create and return an environment
    with the specified parameters.
    """
    def _init():
        env = gym.make(env_id, blue_wrestler=blue_wrestler, red_wrestler=red_wrestler, seed=seed, render_mode=render_mode, resolution=resolution)
        env.metadata['render_fps'] = 24
        return env
    return _init

def final_push(observations):
    x_b, x_r = observations[-1]
    d_b, d_r = x_b, 30-x_r
    f_b = float(d_b - d_r) 
    f_r = -f_b 
    return f_b, f_r

def closest_through_time(o):
    f_r = np.mean(np.argmin(np.abs(o-16), 1))
    f_b = 1-f_r
    return f_b, f_r

def evaluate_wrestlers(vem_fn, fitness_fn, n_steps, frames_samples, tasks, candidates, worker_id=None):
    assert len(tasks) == len(candidates)
    num_envs = len(tasks)
    blue_wrestlers = []
    red_wrestlers = []
    for i in range(len(tasks)):
        if tasks[i]["config"]["generation"] == "red":
            blue_wrestler, red_wrestler = tasks[i]["config"]["wrestler"], candidates[i]["value"]["wrestler"]
        else:
            blue_wrestler, red_wrestler = candidates[i]["value"]["wrestler"], tasks[i]["config"]["wrestler"]
        blue_wrestlers.append(blue_wrestler)
        red_wrestlers.append(red_wrestler)
    env_fns = [make_env('WrestlingEnv-v0', blue_wrestlers[i], red_wrestlers[i], seed=0, render_mode=None) for i in range(num_envs)]
    vec_env = SyncVectorEnv(env_fns) 
    # run 
    #with suppress_output():
    vec_env.reset(seed=0)
    rgb_arrays = np.empty((num_envs, len(frames_samples), 224, 224, 3), dtype=np.uint8)  # CLIP resizes to 224x224 anyway
    Observations = np.empty((num_envs, n_steps, 2))
    Dones = np.zeros(num_envs, dtype=np.bool_)
    for t in range(n_steps):  
        observations, _, dones, truncated, _ = vec_env.step([None] * num_envs)
        Observations[:, t] = observations
        Dones = np.logical_or(Dones, dones)
        if t in frames_samples:
            Quads = vec_env.call("get_quadrilaterals") 
            for i in range(num_envs):
                rgb_arrays[i, frames_samples.index(t)] = get_image(Quads[i])
    # VEM: CLIP
    E = vem_fn(jnp.array(rgb_arrays)/255)  # img shape (H W C) and values in [0, 1].
    # pack evaluations fitness, behavior
    evaluations = []
    for i, (task, candidate) in enumerate(zip(tasks, candidates)):
        if not Dones[i]:  # only save the valid evaluations
            f_b, f_r = fitness_fn(Observations[i])
            if task["config"]["generation"] == "red":  # candidates are red 
                f, other = f_r, f_b
            else:
                f, other = f_b, f_r
            b = np.array(E[i].reshape(-1))
            evaluation = {"task_id": task["id"], "id": None, "solution": candidate["value"], "origin": candidate["origin"], "error": Dones[i],
                          "fitness": f, "behavior": b, "other": other, "obs": Observations[i], "rgb_array": rgb_arrays[i]} 
            evaluations.append(evaluation)    
        else:
            evaluations.append(None)    
    return evaluations


# %% [markdown]
# ### Evaluation for video

# %%
def evaluate_wrestlers_for_video(fitness_fn, n_steps, frames_samples, resolution, tasks, candidates, worker_id=None):
    assert len(tasks) == len(candidates)
    num_envs = len(tasks)
    blue_wrestlers = []
    red_wrestlers = []
    for i in range(len(tasks)):
        if tasks[i]["config"]["generation"] == "red":
            blue_wrestler, red_wrestler = tasks[i]["config"]["wrestler"], candidates[i]["value"]["wrestler"]
        else:
            blue_wrestler, red_wrestler = candidates[i]["value"]["wrestler"], tasks[i]["config"]["wrestler"]
        blue_wrestlers.append(blue_wrestler)
        red_wrestlers.append(red_wrestler)
    with suppress_output():
        env_fns = [make_env('WrestlingEnv-v0', blue_wrestlers[i], red_wrestlers[i], seed=0, render_mode="rgb_array", resolution=resolution) for i in range(num_envs)]
        vec_env = SyncVectorEnv(env_fns) 
        # run 
        vec_env.reset()
        rgb_arrays = np.empty((num_envs, n_steps, resolution[0], resolution[1], 3), dtype=np.uint8)  # CLIP resizes to 224x224 anyway
        Obs = np.empty((num_envs, n_steps, 2)) 
        for t in range(n_steps):  
            next_observations, _, dones, truncated, _ = vec_env.step([None] * num_envs)
            rgb_arrays[:, t] = vec_env.call("render")
            Obs[:, t] = next_observations
            observations = next_observations
        vec_env.close()

    # pack evaluations fitness, behavior
    evaluations = []
    for i, (task, candidate) in enumerate(zip(tasks, candidates)):
        f_b, f_r = fitness_fn(Obs[i])
        if task["config"]["generation"] == "red":  # candidates are red 
            f, other = f_r, f_b
        else:
            f, other = f_b, f_r
        evaluation = {"task_id": task["id"], "id": None, "solution": candidate["value"],
                      "fitness": f, "other": other, "rgb_array": rgb_arrays[i]}
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
            tournament[(i,j)] = {"candidate": {"value":red_team, "origin": i}, "task": {"id": j, "config": blue_team}} 
    batch_size = config["batch_size"]
    to_del = []
    for i in tqdm(range(0, len(tournament), batch_size)):
        keys = list(tournament.keys())[i:i+batch_size]
        if len(keys) < batch_size:
            keys += [keys[-1]]*(batch_size-len(keys))
        candidates = [tournament[key]["candidate"] for key in keys]
        tasks = [tournament[key]["task"] for key in keys]
        evaluations = evaluate_wrestlers_fn(**config["evaluation_config"], tasks=tasks, candidates=candidates)
        for j, key in enumerate(keys):
            if evaluations[j] is not None:
                tournament[key]["eval"] = evaluations[j]
            else:
                to_del.append(key)
    for key in to_del:
        del tournament[key]
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
        f_r, f_b = val["eval"]["fitness"], val["eval"]["other"]
        F[i,j] = f_r
    if sorting:
        red_indices = jnp.argsort(jnp.mean(F, axis=1))
        blue_indices = jnp.argsort(-jnp.mean(F, axis=0))
    plt.subplots(figsize=(10,9))
    plt.pcolor(F[red_indices][:, blue_indices] if sorting else F, vmin=0, vmax=1, cmap="coolwarm")
    plt.xticks([])
    plt.yticks([])
    plt.ylabel("Reds")
    plt.xlabel("Blues")
    plt.colorbar(label="Fitness")
    if path is not None:
        utils.savefig(f"tournament_{gen_id}", path, timestamp=False)
    plt.close()
    return F


# %% [markdown]
# # Config 

# %%
n_steps = 200

wrestlers_config = {
    "frames_indexes": list(np.linspace(0, n_steps-1, 6, dtype=int)[1:]),
    "seed": 0,
    "n_steps": n_steps,
    "width": 5, 
    "num_envs": 20, 
}
fm = CLIP()
vem_fn = jit(vmap(vmap(fm.embed_img)))
#vec_env = init_envs(wrestlers_config['num_envs'], wrestlers_config['width'])
evaluate_wrestlers_fn = partial(evaluate_wrestlers, vem_fn)


# %%
def get_config(wrestlers_config, budget, tasks, n_cells, use_diversity_only=False, use_quality_only=False, seed_id=0):
    config = {  # BT
        "tasks": tasks,
        "budget": budget,
        'batch_size': wrestlers_config['num_envs'],
        'init_elites': np.inf,
        'seed': utils.seeds[seed_id],
        'verbose': True,
        'log_interval': 20_000,
        'use_logging': False,
        "intermediate_videos": False,
        "first_video": False,
        "last_video": False,
        "video_eval_fn": None,
        "avoid_repetition": False,
        "parallel": False,
        "n_proc": N_PROC,
        "archive_config": {
            "use_growing": not use_quality_only,
            'n_cells': n_cells,
            'n_behavior_dim': len(wrestlers_config["frames_indexes"])*512,
            "n_solution_dim": None,
            "use_redristribution": False,
            "use_collection": False,
            "use_repair": True,
            "compare_fitness": (compare_fitness) if not use_diversity_only else compare_fitness_no_quality,
            "distance_function": cosine_distance,
        },
    
        "sample_random_function": sample_random_wrestler,
        "random_sampling_config": {"width": wrestlers_config['width']},
    
        "crossover_and_mutation_function": mutate_wrestler,
        "crossover_and_mutation_config": {'n_mutations': 3, 'mutation_probas': {"deletion": 1/3, "addition": 1/3, "mutation": 1/3}},
        
        "evaluation_function": evaluate_wrestlers_fn, 
        "evaluation_config" : {"fitness_fn": closest_through_time, "n_steps": wrestlers_config["n_steps"], "frames_samples": wrestlers_config["frames_indexes"]},
    }
    return config 


# %% [markdown]
# # Run

# %%
for seed_id in range(3):
    n_tasks = 50  # 50
    n_gen = 10 # 20
    budget = 20_000  # 20_000
    seed_id = 1
    n_cells = 20  # 20
    
    use_boostrap = True
    use_diversity_only = False
    use_quality_only = False
    use_balance_start = False 
    
    if use_balance_start:
        assert use_boostrap
        
    assert not use_diversity_only or not use_quality_only
    mini_config = {"n_tasks": n_tasks, "n_gen": n_gen, "budget": budget, "seed_id": seed_id, "n_cells": n_cells, 
               "use_boostrap": use_boostrap, "use_diversity_only": use_diversity_only, 'use_quality_only': use_quality_only,
              }

    print("main")
    # config 
    config = get_config(wrestlers_config, budget, None, n_cells, use_diversity_only, use_quality_only, seed_id)
    main_folder = utils.create_save_folder()
    utils.save_pickle(main_folder + f"/mini_config.pk", mini_config)
    rng = np.random.default_rng(config["seed"])
    # init
    current_gen = sample_n_random_wrestlers(rng, n_tasks, config['random_sampling_config'])
    bootstrap_evaluations = []
    # main loop
    for gen_id in range(n_gen):
        print(f"Gen {gen_id}: {len(set([get_wrestler_key(wrestler) for wrestler in current_gen]))} different wrestlers.") 
        config["save_folder"] = main_folder + f"/gen_{gen_id}/"
        utils.create_folder(config["save_folder"])
        config["tasks"] = create_tasks(current_gen, generation=["red", "blue"][gen_id%2])  # gen == atk means optimizing the reds and fixing the blues
        me = MT_GAME(config)
        if use_boostrap:
          me.update_archive(bootstrap_evaluations)
        # run gen
        me.run()
        # compute next gen 
        elites = [{"fitness": None, "solution": None} for _ in range(n_tasks)]
        if use_quality_only:
            for task_id, archive in enumerate(me.archive.archives):
                for cell_id in range(config["archive_config"]['n_cells']):
                    if cell_id in archive.non_empty_cells:
                        elites[task_id]["fitness"] = archive.cells_fitness[cell_id]
                        elites[task_id]["solution"] = archive.cells_solution[cell_id]
        else:  # Quality and Diversity
            behaviors = jnp.concatenate([archive.cells_behavior[archive.non_empty_cells] for archive in me.archive.archives])
            kmeans = KMeans(n_clusters=n_tasks).fit(preprocessing.normalize(behaviors))   # for normalized vectors euclidian distance is equivalent to cosine distance ||x-y||² = 2 (1 - x.y)
            centroids = kmeans.cluster_centers_
            tree = cKDTree(centroids)
            elites_solution_txt = [""] * n_tasks
            distances_to_centroids = np.ones(n_tasks) * np.inf
            for task_id, archive in enumerate(me.archive.archives):
                for cell_id in range(config["archive_config"]['n_cells']):
                    if cell_id in archive.non_empty_cells:
                        distance, c_id = tree.query(archive.cells_behavior[cell_id]/jnp.linalg.norm(archive.cells_behavior[cell_id]), k=1)
                        if use_diversity_only:
                            if distance < distances_to_centroids[c_id] and get_wrestler_key(archive.cells_solution[cell_id]) not in elites_solution_txt:
                                distances_to_centroids[c_id] = distance 
                                elites[c_id]["fitness"] = archive.cells_fitness[cell_id]
                                elites[c_id]["solution"] = archive.cells_solution[cell_id]
                                elites_solution_txt[c_id] = get_wrestler_key(archive.cells_solution[cell_id])
                        else:  # diversity and quality 
                            if compare_fitness(archive.cells_fitness[cell_id], elites[c_id]["fitness"]) and get_wrestler_key(archive.cells_solution[cell_id]) not in elites_solution_txt:
                                elites[c_id]["fitness"] = archive.cells_fitness[cell_id]
                                elites[c_id]["solution"] = archive.cells_solution[cell_id]
                                elites_solution_txt[c_id] = get_wrestler_key(archive.cells_solution[cell_id])
        if use_balance_start and gen_id < 2:
            if gen_id == 0:
                first_generations = [elite["solution"] for elite in elites if elite["solution"] is not None]
                utils.save_pickle(config["save_folder"] + f"elites_{gen_id}.pk", {"reds": first_generations, "blues": [task for task in me.tasks]})
                current_gen = sample_n_random_wrestlers(rng, n_tasks, config['random_sampling_config'])
                
            elif gen_id == 1:
                current_gen = [elite["solution"] for elite in elites if elite["solution"] is not None]
                utils.save_pickle(config["save_folder"] + f"elites_{gen_id}.pk", {"reds": [task for task in me.tasks], "blues": current_gen})
                tournament = compute_tournament(config, first_generations, current_gen)
                utils.save_pickle(config["save_folder"] + f"tournament_{gen_id}.pk", tournament)
                bootstrap_evaluations = []
                for (i,j), val in tournament.items():
                    solution = first_generations[i]
                    f = val["eval"]["fitness"]
                    task_id = j
                    bootstrap_evaluations.append({"id": -1, "task_id": task_id, "fitness": f, "behavior": val["eval"]["behavior"], "solution": solution})
                make_tournament_plot(len(first_generations), len(current_gen), tournament, gen_id, config["save_folder"])
        else:
            current_gen = [elite["solution"] for elite in elites if elite["solution"] is not None]
            # compute tournament
            if gen_id % 2 == 1:  # blues 
                blues = current_gen
                reds = [task for task in me.tasks] 
            else:
                reds = current_gen
                blues = [task for task in me.tasks] 
            utils.save_pickle(config["save_folder"] + f"elites_{gen_id}.pk", {"reds": reds, "blues": blues})
            
            # bootstrap next generation
            if use_boostrap:
                tournament = compute_tournament(config, reds, blues)
                #utils.save_pickle(config["save_folder"] + f"tournament_{gen_id}.pk", tournament)
                bootstrap_evaluations = []
                for (i,j), val in tournament.items():
                    if gen_id % 2 == 1:  # blues 
                        solution = reds[i]
                        f = val["eval"]["fitness"]  # red fitness (because the tournament is always red candidates vs blue tasks)
                        task_id = j
                    else:
                        solution = blues[j]
                        f = val["eval"]["other"]  # blue fitness
                        task_id = i 
                    bootstrap_evaluations.append({"id": -1, "task_id": task_id, "fitness": f, "behavior": val["eval"]["behavior"], "solution": solution})
            
                make_tournament_plot(len(reds), len(blues), tournament, gen_id, config["save_folder"])

    Elites = [utils.load_pickle(main_folder + f"/gen_{gen_id}/elites_{gen_id}.pk") for gen_id in range(n_gen)]
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
    
    blues, reds = [], []
    for gen_id in [i for i in range(1, n_gen, 2)]:
        for blue in Elites[gen_id]["blues"]:
            blues.append(blue)
    for gen_id in [i for i in range(0, n_gen, 2)]:
        for red in Elites[gen_id]["reds"]:
            reds.append(red)
    
    config = get_config(wrestlers_config, budget, None, n_cells, use_diversity_only, use_quality_only, seed_id)
    tournament = compute_tournament(config, reds, blues)

    for blue_gen_id, (blue_i, blue_j) in blue_indices.items():
        for red_gen_id, (red_i, red_j) in red_indices.items():
            mini_tournament = {}
            for i in range(red_i, red_j):
                for j in range(blue_i, blue_j):
                    if (i,j) in tournament:
                        mini_tournament[i,j] = tournament[i,j]
            utils.save_pickle(main_folder + f"generational_tournament_{blue_gen_id}_{red_gen_id}.pk", mini_tournament)
