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

#mpl.use('Agg')  # Use non-interactive backend
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
import sys
sys.path.append("hearthbreaker")  
import json
from hearthbreaker.agents.basic_agents import RandomAgent
from hearthbreaker.agents.trade_agent import TradeAgent
from hearthbreaker.cards.heroes import hero_for_class
from hearthbreaker.constants import CHARACTER_CLASS
from hearthbreaker.engine import Game, Deck, card_lookup, card_table
from hearthbreaker.cards import *

# %%
N_PROC = 64


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
        self.parallel_timeout = config["parallel_timeout"]
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
        
    def save_archive(self, name=None):
        archive_save = {"fitness": [], "solutions": [], "behavior": [], "log_id": []}
        for archive in self.archive.archives:
            archive_save["fitness"].append(archive.cells_fitness)
            archive_save["solutions"].append(archive.cells_solution)
            archive_save["behavior"].append(archive.cells_behavior)
            archive_save["log_id"].append(archive.cells_log_id)
        archive_save["log"] = self.log
        archive_save["batch_size"] = self.batch_size
        archive_save["tasks"] = self.tasks

        save_name = self.save_folder + ("/archive_save.pk" if name is None else f"/archive_save_{name}.pk")
        utils.save_pickle(save_name, archive_save) 
        
        
    def sample_candidate(self):
        if self.is_random:  # initialization with random solutions till we find enough elites for diversity
            candidate, origin = self.sample_random(self.rng, self.config["random_sampling_config"])
        else:               
            p1, p2, p1_id, p2_id = self.archive.sample_parents()
            candidate, origin = self.sample_crossover_and_mutation(self.rng, p1, p2, self.config["crossover_and_mutation_config"])
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
        
        self.save_archive("init")
        last_it = self.it
        while self.it < self.budget:
            if self.parallel:  # collect the evaluation
                try:
                    evaluated_candidates = res_queue.get(timeout=self.parallel_timeout)
                except queue.Empty:
                    print("Timeout: workers may be stuck. Restarting.")
                    pool.terminate()
                    pool.join()
                    # Optionally clear the job_queue and refill
                    job_queue = mp.Queue()
                    res_queue = mp.Queue()
                    pool = mp.Pool(self.n_proc, parallel_worker, (self.evaluation_function, job_queue, res_queue))
                    for _ in range(self.n_proc):
                        tasks, candidates = self.sample_new_evaluations()                
                        job_queue.put({"tasks": tasks, "candidates": candidates})
            else:
                tasks, candidates = job_queue.pop(0)
                evaluated_candidates = self.evaluation_function(tasks=tasks, candidates=candidates)
                
            has_changed = self.update_archive(evaluated_candidates)
                                    
            if self.it % self.log_interval == 0:  # save archive 
                #utils.save_pickle(self.save_folder + "/xp.pk", self)
                self.save_archive(self.it)
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
            pool.join()
            
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


# %%
def parallel_worker_with_key(evaluation_function, job_queue, res_queue):
    worker_id = mp.current_process()._identity[0]
    while True:
        try:
            job = job_queue.get(timeout=30)  # Timeout to check for shutdown
            if job is None:
                break
            key, args = job
            try:
                res = evaluation_function(tasks=args["tasks"], candidates=args["candidates"], worker_id=worker_id)
                res_queue.put((key, res), timeout=10)
            except:
                print(f"Worker error processing {key}: {e}")
                res_queue.put((key, [None]), timeout=10)
        except queue.Empty:
            # Periodic check for shutdown - could check a shared flag here
            continue
        except Exception as e:
            print(f"Worker exception: {e}")
            break


# %% [markdown]
# ## Hearthstone ENV

# %% [markdown]
# ### Fitness

# %%
def compare_fitness(new, old):
    return (old is None) or (new > old)

def compare_fitness_no_quality(new, old):
    return (old is None)

def euclidiane_distance(x, Y):
    return np.linalg.norm(x-Y, axis=1)

def cosine_distance(x, Y):
    dot_product = np.dot(Y, x)
    norm_x = np.linalg.norm(x)
    norm_Y = np.linalg.norm(Y, axis=1)
    return 1-dot_product/(norm_x * norm_Y)


# %% [markdown]
# ### Sampling

# %%
def sample_random_deck(rng, config):
    hero_class = config["hero_class"]
    assert hero_class != 0 and hero_class in valid_cards_names
    deck = rng.choice(valid_cards_names[hero_class], 30)
    return {"deck": deck, "hero_class": hero_class}, "random"


# %%
def get_deck_key(solution):
    l = list(solution["deck"])
    l.sort()
    return "_".join(l)
    
def sample_n_random_decks(rng, n, config):
    initial_blues = []
    keys = set()
    while len(keys) < n:
        sol, _ = sample_random_deck(rng, config)
        key = get_deck_key(sol)
        if key not in keys:
            keys.add(key)
            initial_blues.append(sol)
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
def mutate_deck(rng, sol, _, __):  
    """
    second parent ignored at the moment (i.e., no crossover)
    do not have config etheir
    """
    hero_class = sol["hero_class"]
    deck = np.copy(sol["deck"])
    k = 31
    while k>30:
        k = rng.geometric(0.5)  # follow Fontaine et al. (2019) by using a geometric distribution for the number of card changed
    old_cards = rng.choice([i for i in range(30)], k)
    for old_card_id in old_cards:
        deck[old_card_id] = ""
        while deck[old_card_id] == "":
            new_card = rng.choice(valid_cards_names[hero_class])
            if np.sum(deck == new_card) < 2:  # check that deck is valid
                deck[old_card_id] = new_card
    return {"deck": deck, "hero_class": hero_class}, {"type": "mutation", "k": k}


# %% [markdown]
# ### Evaluation

# %%
def get_behavior(deck):
    mana = [c.mana for c in deck.cards]
    return np.array([np.mean(mana), np.std(mana)])

def write_deck(path, deck_list):
    with open(path, "w") as f:
        f.write("1 ")
        f.writelines("\n1 ".join(deck_list))     

def evaluate_hearthstone_subproc(agent_name, n_replications, seed, exec_path, timeout, tasks, candidates, worker_id=None):
    path = f"/home/tim/Experiments/data/GAME/Hearthstone/tmp/{worker_id}_" 
    write_deck(path+"task_deck.txt", tasks[0]["config"]["deck"])
    write_deck(path+"candidate_deck.txt", candidates[0]["value"]["deck"])
    args = (path, str(n_replications), str(seed), str(tasks[0]["config"]["hero_class"]), str(candidates[0]["value"]["hero_class"]), tasks[0]["config"]["generation"], agent_name)
    with open(path+"test.txt", "w") as f:
        f.writelines("\n".join(list(args)))  
    try:
        # Run the subprocess and capture output
        result = subprocess.run(
            [sys.executable, exec_path] + list(args),
            check=True,
            timeout=timeout,  # Timeout in seconds
        )
    except subprocess.CalledProcessError as e:
        return [None]
    except subprocess.TimeoutExpired:
        return [None]
    data = utils.load_pickle(path+"evaluation.pk")  # TODO something when the data is None !!!!
    if data is None:
        return [None]
        
    # pack evaluations fitness, behavior
    evaluations = []
    f_r = data["f_r"]
    f_b = -f_r
    if tasks[0]["config"]["generation"] == "red":  # candidates are red 
        f, other_f = f_r, f_b
        b = data["red_behavior"]
        other_b = data["blue_behavior"]
    else:
        f, other_f = f_b, f_r
        b = data["blue_behavior"]
        other_b = data["red_behavior"]
    
    evaluation = {"task_id": tasks[0]["id"], "id": None, "solution": candidates[0]["value"], "origin": candidates[0]["origin"], 
                  "fitness": f, "behavior": b, "other_fitness": other_f, "other_behavior": other_b} 
    evaluations.append(evaluation)    
    
    return evaluations
    

def evaluate_hearthstone(agent, n_replications, tasks, candidates, worker_id=None):  # made for multi-proc not batch 
    task_deck = Deck([valid_cards[c]() for c in tasks[0]["config"]["deck"]], hero_for_class(tasks[0]["config"]["hero_class"]))
    candidate_deck = Deck([valid_cards[c]() for c in candidates[0]["value"]["deck"]], hero_for_class(candidates[0]["value"]["hero_class"]))
    if tasks[0]["config"]["generation"] == "red":
        blue_deck, red_deck = task_deck, candidate_deck
    else:
        blue_deck, red_deck = candidate_deck, task_deck

    # run 
    H_diff = np.zeros(n_replications)
    game = Game([red_deck, blue_deck], [agent, agent])
    try:
        for rep in range(n_replications):
            new_game = game.copy()
            new_game.start()
            h_red, h_blue = new_game.players[0].hero.health, new_game.players[1].hero.health
            del new_game
            H_diff[rep] = h_red - h_blue  # like Fontaine 2019, fitness = health one - health of the other 
    except:
        #print(tasks, candidates)
        return [None]
    del game 

    
    # pack evaluations fitness, behavior
    evaluations = []
    f_r = np.mean(H_diff)  
    f_b = -f_r
    if tasks[0]["config"]["generation"] == "red":  # candidates are red 
        f, other_f = f_r, f_b
        b = get_behavior(red_deck)
        other_b = get_behavior(blue_deck)
    else:
        f, other_f = f_b, f_r
        b = get_behavior(blue_deck)
        other_b = get_behavior(red_deck)
    
    evaluation = {"task_id": tasks[0]["id"], "id": None, "solution": candidates[0]["value"], "origin": candidates[0]["origin"], 
                  "fitness": f, "behavior": b, "other_fitness": other_f, "other_behavior": other_b} 
    evaluations.append(evaluation)    
    return evaluations

def fake_evaluate_hearthstone(agent, n_replications, tasks, candidates, worker_id=None):  # made for multi-proc not batch 
    task_deck = Deck([valid_cards[c]() for c in tasks[0]["config"]["deck"]], hero_for_class(tasks[0]["config"]["hero_class"]))
    candidate_deck = Deck([valid_cards[c]() for c in candidates[0]["value"]["deck"]], hero_for_class(candidates[0]["value"]["hero_class"]))
    if tasks[0]["config"]["generation"] == "red":
        blue_deck, red_deck = task_deck, candidate_deck
    else:
        blue_deck, red_deck = candidate_deck, task_deck

    # run 
    H_diff = np.zeros(n_replications)
   
    try:
        for rep in range(n_replications):
            h_red, h_blue = np.random.randint(30), np.random.randint(30)
            H_diff[rep] = h_red - h_blue  # like Fontaine 2019, fitness = health one - health of the other 
    except:
        #print(tasks, candidates)
        return [None]

    # pack evaluations fitness, behavior
    evaluations = []
    f_r = np.mean(H_diff)  
    f_b = -f_r
    if tasks[0]["config"]["generation"] == "red":  # candidates are red 
        f, other_f = f_r, f_b
        b = get_behavior(red_deck)
        other_b = get_behavior(blue_deck)
    else:
        f, other_f = f_b, f_r
        b = get_behavior(blue_deck)
        other_b = get_behavior(red_deck)
    
    evaluation = {"task_id": tasks[0]["id"], "id": None, "solution": candidates[0]["value"], "origin": candidates[0]["origin"], 
                  "fitness": f, "behavior": b, "other_fitness": other_f, "other_behavior": other_b} 
    evaluations.append(evaluation)    
    return evaluations


# %% [markdown]
# ## Tournament

# %%
import threading
import time


# %%
def pool_join_with_timeout(pool, timeout=30):
    """Join pool with timeout using threading"""
    def join_target():
        pool.join()
    
    join_thread = threading.Thread(target=join_target)
    join_thread.daemon = True
    join_thread.start()
    join_thread.join(timeout)
    
    return not join_thread.is_alive()

def compute_tournament(config, reds, blues, max_concurrent=None):
    """Process tournament continuously with better timeout handling"""
    n_proc = min(N_PROC, len(reds)*len(blues))
    
    if max_concurrent is None:
        max_concurrent = n_proc
    
    job_queue = mp.Queue(maxsize=max_concurrent)  # using manager next time could make shuting down cleaner
    res_queue = mp.Queue()
    
    pool = mp.Pool(n_proc, parallel_worker_with_key, 
                   (partial(config["evaluation_function"], **config["evaluation_config"]), 
                    job_queue, res_queue))
    
    tournament = {}
    all_pairs = []
    collected_pairs = {}
    # Generate all pairs
    for i, red_team in enumerate(reds):
        for j, blue_team in enumerate(blues):
            blue_team["generation"] = "red"
            task = {"id": j, "config": blue_team}
            candidate = {"value": red_team, "origin": i}
            all_pairs.append((i, j))
            collected_pairs[(i,j)] = False
            tournament[(i,j)] = {"candidate": candidate, "task": task}
            
    total_pairs = len(all_pairs)
    jobs_submitted = 0
    results_collected = 0
    consecutive_timeouts = 0
    max_consecutive_timeouts = 3
    
    with tqdm(total=total_pairs, desc="Tournament") as pbar:
        # Submit initial batch
        while jobs_submitted < min(max_concurrent, total_pairs):
            key = all_pairs[jobs_submitted]
            job_queue.put((key, {"tasks": [tournament[key]["task"]], "candidates": [tournament[key]["candidate"]]}))
            jobs_submitted += 1
        
        jobs_submitted = jobs_submitted % total_pairs
        
        # Continuous processing
        while results_collected < total_pairs and consecutive_timeouts < max_consecutive_timeouts:
            try:
                # Use shorter timeout but don't give up immediately
                key, evaluations = res_queue.get(timeout=30)
                ev = evaluations[0]
                consecutive_timeouts = 0  # Reset timeout counter
                
                if key in tournament and not collected_pairs[key]:  # make sure not to collect the same pair twice
                    tournament[key]["eval"] = ev
                    results_collected += 1
                    pbar.update(1)
                    collected_pairs[key] = True
                
                # Submit next job if available
                if results_collected < total_pairs:
                    try:
                        key = all_pairs[jobs_submitted]
                        job_queue.put((key, {"tasks": [tournament[key]["task"]], "candidates": [tournament[key]["candidate"]]}), timeout=0.1)
                        jobs_submitted = (jobs_submitted + 1) % total_pairs
                        while collected_pairs[all_pairs[jobs_submitted]]: # there shouldn't be any infinite loop because np.sum(collected_pairs) == results_collected < total_pairs
                            jobs_submitted = (jobs_submitted + 1) % total_pairs
                        
                    except queue.Full:
                        pass  # Will try again next iteration
                        
            except queue.Empty:
                consecutive_timeouts += 1
                print(f"Timeout #{consecutive_timeouts}. Progress: {results_collected}/{total_pairs}")
                
                # If we've submitted all jobs but haven't collected all results,
                # and we're getting timeouts, some jobs might be stuck
                if jobs_submitted == total_pairs:
                    print("All jobs submitted, waiting for remaining results...")
                    # Give more time for remaining jobs
                    if consecutive_timeouts >= max_consecutive_timeouts:
                        print("Too many consecutive timeouts, stopping collection")
                        break
    
    # Clean up failed evaluations
    to_del = [key for key, val in tournament.items() if val["eval"] is None]
    
    for key in to_del:
        del tournament[key]

    # empty job queue
    try:
        while True:
            try:
                job_queue.get_nowait()  # Non-blocking get
                drained_count += 1
            except queue.Empty:
                break
    except Exception as e:
        print(f"Error draining queue: {e}")
        
    # Send sentinel values to stop workers
    for _ in range(n_proc):
        try:
            job_queue.put(None, timeout=2)
        except queue.Full:
            pass
                
    # Quick fix - add timeouts to cleanup
    job_queue.close()
    res_queue.close()
    pool.terminate()
    
    if not pool_join_with_timeout(pool, timeout=30):
        print("Pool didn't terminate cleanly, killing processes")
        for process in pool._pool:
            if process.is_alive():
                process.kill()
        
        # Give kills time to take effect, then final join
        time.sleep(1)
        try:
            pool.join()  # Should be quick now
        except:
            pass
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
def make_tournament_plot(n_red, n_blue, tournament, gen_id, path, sorting=True):
    F = np.zeros((n_red, n_blue))
    for (i,j), val in tournament.items():
        f_r, f_b = val["eval"]["fitness"], val["eval"]["other_fitness"]
        F[i,j] = f_r
    if sorting:
        red_indices = np.argsort(np.mean(F, axis=1))
        blue_indices = np.argsort(-np.mean(F, axis=0))
    plt.subplots(figsize=(10,9))
    plt.pcolor(F[red_indices][:, blue_indices] if sorting else F, vmin=-30, vmax=30, cmap="coolwarm")
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
classes_names = {
    0: "Common",
    1: "Mage",
    2: "Hunter",
    3: "Shaman",
    4: "Warrior",
    5: "Druid",
    6: "Priest", 
    7: "Paladin",
    8: "Rogue",
    9: "Warlock",
}

mage_starting_deck = ["Arcane Missiles", "Murloc Raider", "Arcane Explosion",
                        "Bloodfen Raptor", "Novice Engineer", "River Crocolisk", 
                        "Arcane Intellect", "Raid Leader", "Wolfrider", "Fireball",
                        "Oasis Snapjaw", "Polymorph", "Sen'jin Shieldmasta", 
                        "Nightblade", "Boulderfist Ogre"] * 2 

hunter_starting_deck = ["Arcane Shot", "Stonetusk Boar", "Timber Wolf",
                        "Tracking", "Bloodfen Raptor", "River Crocolisk",
                        "Ironforge Rifleman", "Raid Leader", "Razorfen Hunter", 
                        "Silverback Patriarch", "Houndmaster", "Multi-Shot", 
                        "Oasis Snapjaw", "Stormpike Commando", "Core Hound"] * 2

druid_starting_deck = ["Innervate", "Claw", "Elven Archer",
                        "Mark of the Wild", "River Crocolisk", "Wild Growth",
                        "Healing Touch", "Silverback Patriarch", "Chillwind Yeti",
                        "Oasis Snapjaw", "Darkscale Healer", "Nightblade",
                        "Boulderfist Ogre", "Lord of the Arena", "Core Hound"]*2

paladin_starting_deck = ['Blessing of Might', 'Goldshire Footman', 'Hand of Protection', "Light's Justice", 'Stonetusk Boar', 'Holy Light', 'Ironforge Rifleman', 
                         'Raid Leader', 'Gnomish Inventor', 'Hammer of Wrath', 'Stormwind Knight', 'Nightblade', 'Stormpike Commando', 'Lord of the Arena', 'Stormwind Champion']*2

priest_starting_deck = ['Elven Archer', 'Holy Smite', 'Northshire Cleric', 'Power Word: Shield', 'Voodoo Doctor', 'Bloodfen Raptor', 'Frostwolf Grunt', 'Mind Blast', 'Shadow Word: Pain',  # Mind Blast -> Radiance later on
                        'Shattered Sun Cleric', 'Silverback Patriarch', 'Chillwind Yeti', "Sen'jin Shieldmasta", 'Gurubashi Berserker', 'Core Hound'] * 2

rogue_starting_deck = ['Backstab', 'Deadly Poison', 'Elven Archer', 'Goldshire Footman', 'Sinister Strike', 'Bloodfen Raptor', 'Novice Engineer', 'Sap', 'Ironforge Rifleman', 
                       'Dragonling Mechanic', 'Gnomish Inventor', 'Stormwind Knight', 'Assassinate', 'Nightblade', 'Stormpike Commando']*2

shaman_starting_deck = ['Ancestral Healing', 'Frost Shock', 'Stonetusk Boar', 'Frostwolf Grunt', 'Rockbiter Weapon', 'Windfury', 'Raid Leader', 'Wolfrider', 
                        'Chillwind Yeti', 'Hex', "Sen'jin Shieldmasta", 'Booty Bay Bodyguard', 'Frostwolf Warlord', 'Boulderfist Ogre', 'Reckless Rocketeer']*2

warlock_starting_deck = ['Murloc Raider', 'Voidwalker', 'Voodoo Doctor', 'Kobold Geomancer', 'River Crocolisk', 'Succubus', 'Drain Life', 'Shadow Bolt', 'Wolfrider', # Succubes renamed Felwalker later on
                        'Chillwind Yeti', 'Hellfire', 'Ogre Magi', 'Darkscale Healer', 'Reckless Rocketeer', 'War Golem']*2

warrior_starting_deck = ['Charge', 'Murloc Raider', 'Execute', 'Frostwolf Grunt', 'Heroic Strike', 'Murloc Tidehunter', 'Fiery War Axe', 'Razorfen Hunter',
                        'Warsong Commander', 'Wolfrider', 'Dragonling Mechanic', "Sen'jin Shieldmasta", 'Gurubashi Berserker', 'Boulderfist Ogre', 'Lord of the Arena']*2

classes_starting_deck = {
    1: mage_starting_deck,
    2: hunter_starting_deck,
    3: shaman_starting_deck,
    4: warrior_starting_deck,
    5: druid_starting_deck,
    6: priest_starting_deck, 
    7: paladin_starting_deck,
    8: rogue_starting_deck,
    9: warlock_starting_deck,
}

# %%
hearthstone_config = {
    "blue_hero": CHARACTER_CLASS.WARRIOR,
    "red_hero": CHARACTER_CLASS.HUNTER,
    "n_replications": 50,
    "agent": "Trader",
    "timeout": 30,
    "exec_path":"/PATH/TO/GAME/Hearthbreaker/run_games.py",
}

ban_list = [
    # Mage
   "Ice Block",  # the immune buff seems to create an infinite loop with the TradeAgent (seems to be catch by the recursion fix, expected but to be safe I removed it)
   "Echo of Medivh",  # something with minion copy havign None as player attribute (seems to be catch by the recursion fix, weird??)
    # Hunter
   'Glaivezooka',  # something with battlcry not being iterable
    # Druid
    'Recycle', 
    # Paladin
    'Coghammer',
    'Argent Lance',
    # Rogue
    "Perdition's Blade",
]

valid_card_name_per_class = {}
valid_cards = {}
for name, _card in card_table.items():
    if name not in ban_list:
        card = _card()
        if card.collectible:
            if card.character_class not in valid_card_name_per_class:
                valid_card_name_per_class[card.character_class] = []
            valid_card_name_per_class[card.character_class].append(name) 
            valid_cards[name] = _card 

valid_cards_names = {i: (valid_card_name_per_class[0] + valid_card_name_per_class[i])*2 for i in range(1, 10)}


# %%
def get_config(hearthstone_config, budget, tasks, n_cells, side, use_diversity_only=False, use_quality_only=False, seed_id=0):
    assert side in ["red", "blue"]
    config = {  # BT
        "tasks": tasks,
        "budget": budget,
        'batch_size': 1,
        'init_elites': 100,
        'seed': utils.seeds[seed_id],
        'verbose': True,
        'log_interval': 10_000,
        'use_logging': False,
        "intermediate_videos": False,
        "first_video": False,
        "last_video": False,
        "video_eval_fn": None,
        "avoid_repetition": False,
        "parallel": True,
        "parallel_timeout": 60,
        "n_proc": N_PROC,
        "archive_config": {
            "use_growing": not use_quality_only,
            'n_cells': n_cells,
            'n_behavior_dim': 2,
            "n_solution_dim": None,
            "use_redristribution": False,
            "use_collection": False,
            "use_repair": True,
            "compare_fitness": (compare_fitness) if not use_diversity_only else compare_fitness_no_quality,
            "distance_function": euclidiane_distance,
        },
    
        "sample_random_function": sample_random_deck,
        "random_sampling_config": {"hero_class": hearthstone_config["blue_hero"] if side == "blue" else hearthstone_config["red_hero"]},
    
        "crossover_and_mutation_function": mutate_deck,
        "crossover_and_mutation_config": {},
        
        "evaluation_function": evaluate_hearthstone_subproc, 
        #"evaluation_config" : {"agent": hearthstone_config["agent"], "n_replications": hearthstone_config["n_replications"]},
        "evaluation_config" : {"agent_name": hearthstone_config["agent"], "n_replications": hearthstone_config["n_replications"], "seed": utils.seeds[seed_id],
                                "exec_path": hearthstone_config["exec_path"], "timeout": hearthstone_config["timeout"]},
    }
    return config 


# %% [markdown]
# # Run

# %%
if __name__ == "__main__":
    Classes = [
        (CHARACTER_CLASS.WARRIOR, CHARACTER_CLASS.WARLOCK),
        (CHARACTER_CLASS.ROGUE, CHARACTER_CLASS.PALADIN),
        (CHARACTER_CLASS.HUNTER, CHARACTER_CLASS.DRUID),
        (CHARACTER_CLASS.SHAMAN, CHARACTER_CLASS.MAGE),
        (CHARACTER_CLASS.MAGE, CHARACTER_CLASS.PRIEST),
    ]
    for (red_hero, blue_hero) in Classes:
        hearthstone_config["red_hero"] = red_hero
        hearthstone_config["blue_hero"] = blue_hero
        
        print(classes_names[red_hero], " VS ", classes_names[blue_hero])
        n_tasks = 50  # 50
        n_gen = 8 # 8
        budget = 10_000  # 10_000
        seed_id = 0
        n_cells = 20 # 20
        
        use_boostrap = True  # True 
        use_diversity_only = False
        use_quality_only = False
        use_balance_start = False 
        with_tournament_plot = False
        use_inter_generational_tournament = False  # True
        
        if use_balance_start:
            assert use_boostrap
            
        assert not use_diversity_only or not use_quality_only
        mini_config = {"n_tasks": n_tasks, "n_gen": n_gen, "budget": budget, "seed_id": seed_id, "n_cells": n_cells, "hearthstone_config": hearthstone_config,
                   "use_boostrap": use_boostrap, "use_diversity_only": use_diversity_only, 'use_quality_only': use_quality_only,
                  }
        print("main")
        # config 
        config = get_config(hearthstone_config, budget, None, n_cells, "red", use_diversity_only, use_quality_only, seed_id)
        main_folder = utils.create_save_folder()
        utils.save_pickle(main_folder + f"/mini_config.pk", mini_config)
        rng = np.random.default_rng(config["seed"])
        # init
        current_gen = sample_n_random_decks(rng, n_tasks, {"hero_class": hearthstone_config["blue_hero"]})
        bootstrap_evaluations = []
        # main loopnt
        for gen_id in range(n_gen):
            print(f"Gen {gen_id}: {len(set([get_deck_key(sol) for sol in current_gen]))} different tasks.") 
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
                            elites[task_id]["solution"] = archive.cells_solution[celqueue.Empty:l_id]
            else:  # Quality and Diversity
                behaviors = np.concatenate([archive.cells_behavior[archive.non_empty_cells] for archive in me.archive.archives])
                print(f"{len(set([tuple(b) for b in behaviors]))} behaviors found")
                kmeans = KMeans(n_clusters=n_tasks).fit(preprocessing.normalize(behaviors))   # for normalized vectors euclidian distance is equivalent to cosine distance ||x-y||² = 2 (1 - x.y)
                centroids = kmeans.cluster_centers_
                tree = cKDTree(centroids)
                elites_solution_txt = [""] * n_tasks
                distances_to_centroids = np.ones(n_tasks) * np.inf
                for task_id, archive in enumerate(me.archive.archives):
                    for cell_id in range(config["archive_config"]['n_cells']):
                        if cell_id in archive.non_empty_cells:
                            distance, c_id = tree.query(archive.cells_behavior[cell_id]/np.linalg.norm(archive.cells_behavior[cell_id]), k=1)
                            if use_diversity_only:
                                if distance < distances_to_centroids[c_id] and get_deck_key(archive.cells_solution[cell_id]) not in elites_solution_txt:
                                    distances_to_centroids[c_id] = distance 
                                    elites[c_id]["fitness"] = archive.cells_fitness[cell_id]
                                    elites[c_id]["solution"] = archive.cells_solution[cell_id]
                                    elites_solution_txt[c_id] = get_deck_key(archive.cells_solution[cell_id])
                            else:  # diversity and quality 
                                if compare_fitness(archive.cells_fitness[cell_id], elites[c_id]["fitness"]) and get_deck_key(archive.cells_solution[cell_id]) not in elites_solution_txt:
                                    elites[c_id]["fitness"] = archive.cells_fitness[cell_id]
                                    elites[c_id]["solution"] = archive.cells_solution[cell_id]
                                    elites_solution_txt[c_id] = get_deck_key(archive.cells_solution[cell_id])
            if use_balance_start and gen_id < 2:
                if gen_id == 0:
                    first_generations = [elite["solution"] for elite in elites if elite["solution"] is not None]
                    utils.save_pickle(config["save_folder"] + f"elites_{gen_id}.pk", {"reds": first_generations, "blues": [task for task in me.tasks]})
                    current_gen = sample_n_random_decks(rng, n_tasks, {"hero_class": hearthstone_config["red_hero"]})
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
                    if with_tournament_plot:
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
                    utils.save_pickle(config["save_folder"] + f"tournament_{gen_id}.pk", tournament)
                    bootstrap_evaluations = []
                    for (i,j), val in tournament.items():
                        if gen_id % 2 == 1:  # blues 
                            solution = reds[i]
                            f = val["eval"]["fitness"]  # red fitness (because the tournament is always red candidates vs blue tasks)
                            b = val["eval"]["behavior"]
                            task_id = j
                        else:
                            solution = blues[j]
                            f = val["eval"]["other_fitness"]  # blue fitness
                            b = val["eval"]["other_behavior"]
                            task_id = i 
                        bootstrap_evaluations.append({"id": -1, "task_id": task_id, "fitness": f, "behavior": b, "solution": solution})
                    if with_tournament_plot:
                        make_tournament_plot(len(reds), len(blues), tournament, gen_id, config["save_folder"])
        
        if use_inter_generational_tournament:
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
            
            config = get_config(hearthstone_config, budget, None, n_cells, "red", use_diversity_only, use_quality_only, seed_id)
            tournament = compute_tournament(config, reds, blues)
        
            for blue_gen_id, (blue_i, blue_j) in blue_indices.items():
                for red_gen_id, (red_i, red_j) in red_indices.items():
                    mini_tournament = {}
                    for i in range(red_i, red_j):
                        for j in range(blue_i, blue_j):
                            if (i,j) in tournament:
                                mini_tournament[i,j] = tournament[i,j]
                    utils.save_pickle(main_folder + f"generational_tournament_{blue_gen_id}_{red_gen_id}.pk", mini_tournament)

# %%
