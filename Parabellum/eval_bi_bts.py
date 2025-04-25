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
import os

# %%
import sys
sys.path.append("..")  # import utils from parent folder 
import utils 

# %%
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from jax import vmap, random
from jax.lax import fori_loop
import jax.numpy as jnp
from flax.struct import dataclass
import chex

# %%
from clip import CLIP
fm = CLIP()

# %%
from atomics import *
from BT_array import *
from BT_tree import * 
from env import *
from grammar import * 


# %% [markdown]
# # Env

# %% [markdown]
# ## Functions

# %% [markdown]
# ### spawn function

# %%
def spawn_fn_factory(env):
    indices = jnp.indices((env.size, env.size)).reshape(2, -1).T  # suppose they all have the same shape which should be true!
    """Spawns n agents on a map."""
    def spawn_one(rng, spawning_sector):
        key_start, key_noise = random.split(rng, 2)
        noise = 0.25 + random.uniform(key_noise, (2,)) * 0.5
        idx = indices[random.categorical(key_start, jnp.log(spawning_sector.reshape(-1)))]
        return idx + noise
    vmap_agents = vmap(spawn_one)
    vmap_bts = vmap(vmap_agents)
    return vmap_bts


# %%
def init_state_factory(init_rng, env, n_bts):
    init_rng = random.split(init_rng, env.num_agents)
    init_rng = repeat(init_rng, "agents keys -> y agents keys", y=n_bts)  # copy rng for bts dimensions
    spawn_fn = spawn_fn_factory(env)
    vmap_bts_compute_unit_in_sight_distance = vmap(compute_unit_in_sight_distance, in_axes=(None, None, 0)) # vmap on the bts
    def init_state_fn(scenario):
        state = State(
            unit_positions = spawn_fn(init_rng, scenario.unit_starting_sectors),
            unit_health = env.unit_type_health[scenario.unit_types],
            unit_cooldowns = jnp.zeros((n_bts, env.num_agents), dtype=jnp.int32),
            unit_in_sight_distance = jnp.ones((n_bts, env.num_agents, env.num_agents)),
            time = jnp.zeros(n_bts, dtype=jnp.int32),
        )    
        state = replace(state, unit_in_sight_distance=vmap_bts_compute_unit_in_sight_distance(env, scenario, state))
        return state
    return init_state_fn


# %% [markdown]
# ### make_init_scenario

# %%
def make_init_scenario(env, max_map_targets, n_bts, places, movement_randomness = 5., units_push_back_firmness = 0.02,):
    # terrains 
    terrains = []
    for place in places:
        if place in terrain_db.db:
            terrain = terrain_db.make_terrain(terrain_db.db[place], env.size)
        elif place in terrain_db.saved_data:
            terrain = terrain_db.load_data(place)
        elif place in terrain_db.homemade_maps:
            terrain = terrain_db.load_homemade(place, env.size)
        else:
            terrain = geo.geography_fn(place, env.size)
        terrains.append(terrain)

    unit_starting_sectors = jnp.zeros((n_bts, env.num_agents, env.size, env.size), dtype=jnp.int32)
    unit_types = jnp.zeros((n_bts, env.num_agents), dtype=jnp.uint8)
    unit_target_position_id = jnp.zeros((n_bts, env.num_agents), dtype=jnp.uint32)
    unit_team = jnp.array([[1]*env.num_allies+[0]*env.num_enemies]*n_bts)
    distance_map = jnp.zeros((n_bts, max_map_targets, env.size, env.size), dtype=jnp.uint32)
    targets_position = -jnp.ones((n_bts, max_map_targets, 2))  # negative values mean not set 

    return Scenario(
        unit_types = unit_types,
        unit_team = unit_team,
        unit_target_position_id= unit_target_position_id,
        unit_starting_sectors = unit_starting_sectors,
        terrain = stack_terrain(terrains),
        distance_map = distance_map,
        movement_randomness = jnp.array([movement_randomness]*n_bts),
        units_push_back_firmness = jnp.array([units_push_back_firmness]*n_bts),
        targets_position = targets_position
    )


# %% [markdown]
# ### step factory

# %%
def step_factory(env, steps_rng, n_steps, all_variants, bt_max_size):
    
    get_action = get_action_factory(all_variants, env.num_agents, bt_max_size)
    #@scan_tqdm(n_steps)  # need to pass an array of int 
    def step(carry, i):
        scenario, state, predecessors, parents, passing_nodes, variant_ids = carry 
        action_rng, step_rng = steps_rng[i]
        # compute the actions 
        vmap_agents_get_action = vmap(get_action, in_axes=(None, None, None, 0, 0, 0,  0, 0, 0))
        vmap_bts_get_action = vmap(vmap_agents_get_action, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, None))
        actions, actions_id = vmap_bts_get_action(env, scenario, state, action_rng, predecessors, parents, passing_nodes, variant_ids, jnp.arange(env.num_agents))
        # step in the env 
        vmap_bts_step = vmap(env_step, in_axes=(None, 0, 0, 0, 0)) # vmap on the bts
        state, scenario = vmap_bts_step(env, scenario, state, actions, step_rng) 
        carry = (scenario, state, predecessors, parents, passing_nodes, variant_ids)
        return carry, (state, actions, actions_id)
        
    return step


# %% [markdown]
# ### plot

# %%
_allies_colors = jnp.array([[0, 0, 128], [65, 105, 225], [135, 206, 235], [0, 255, 127], [34, 139, 34]])/255
_allies_trees_colors = jnp.array([[0, 72, 88], [65, 143, 167], [135, 216, 205], [60, 189, 113], [1, 68, 33]])/255
_enemies_colors = jnp.array([[178, 34, 34], [220, 20, 60], [255, 99, 71], [255, 192, 203], [238, 130, 238]])/255
_enemies_trees_colors = jnp.array([[148, 64, 34], [180, 50, 50], [225, 129, 81], [225, 222, 203], [142, 58, 89]])/255
_mixed_team = jnp.array([[190, 0, 190]])/255
_mixed_team_trees = jnp.array([[150, 40, 160]])/255

def get_rbg_array(indexes, state_seq, env, scenario,  bts_ids, units_resolution=100):
    compute_image = plot_step_3d_factory(scenario, state_seq, env, units_resolution)
    frames = vmap(vmap(compute_image, in_axes=(0, None)), in_axes=(None, 0))(indexes, bts_ids)  
    return frames

def plot_step_3d_factory(scenario, state_seq, env, resolution, 
                        allies_colors=_allies_colors, allies_trees_colors=_allies_trees_colors, 
                        enemies_colors=_enemies_colors, enemies_trees_colors=_enemies_trees_colors,
                        mixed_team=_mixed_team, mixed_team_trees=_mixed_team_trees):
    scale = resolution/env.size
    units_image = jnp.ones((resolution, resolution, 3))
    def plot_step(t, bts_id):
        allies_colors_ = allies_colors[scenario.unit_types[bts_id, :env.num_allies]]
        allies_trees_colors_ = allies_trees_colors[scenario.unit_types[bts_id, :env.num_allies]]
        enemies_colors_ = enemies_colors[scenario.unit_types[bts_id, env.num_allies:]]
        enemies_trees_colors_ = enemies_trees_colors[scenario.unit_types[bts_id, env.num_allies:]]
        colors = jnp.concatenate([allies_colors_, enemies_colors_])
        colors_trees = jnp.concatenate([allies_trees_colors_, enemies_trees_colors_])
        idx = (t,) +  (bts_id,)
        indices = jnp.array(state_seq.unit_positions[idx] * scale, dtype=jnp.uint32)
        in_forest_indices = jnp.array(state_seq.unit_positions[idx], dtype=jnp.uint32)
        in_forest = scenario.terrain.forest[bts_id, in_forest_indices[:,0], in_forest_indices[:, 1]].reshape(env.num_agents, 1)   
        units_alive = (state_seq.unit_health[idx] > 0).reshape(env.num_agents, 1)   
        new_value = jnp.where(units_alive, jnp.where(in_forest, colors_trees, colors), units_image[indices[:, 0], indices[:, 1]])
        image = jax.lax.fori_loop(0, env.num_agents,  lambda i, base: base.at[indices[i,0], indices[i,1]].set(new_value[i]), units_image)
        allies_pos = jnp.zeros(units_image.shape, dtype=jnp.bool_).at[indices[:env.num_allies, 0], indices[:env.num_allies, 1]].set(jnp.where(units_alive[:env.num_allies], True, False))
        enemies_pos = jnp.zeros(units_image.shape, dtype=jnp.bool_).at[indices[env.num_allies:, 0], indices[env.num_allies:, 1]].set(jnp.where(units_alive[env.num_allies:], True, False))
        image = jnp.where(jnp.logical_and(allies_pos, enemies_pos), mixed_team, image)
        return image
    return jit(plot_step)


# %% [markdown]
# ### get video

# %%
def get_basemap_array(scenario, bts_id, resolution, figsize=(10,10)):
    fig, ax = plt.subplots(figsize=figsize)
    plt.imshow(scenario.terrain.basemap[bts_id], extent=(0, resolution, 0, resolution))
    plt.xlim((0, resolution))
    plt.ylim((0, resolution))
    plt.axis("off")
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    basemap_image = jnp.array(fig.canvas.renderer._renderer, dtype=jnp.uint32)  # type: ignore
    plt.close(fig)
    return basemap_image
    
def plot_episode_jax(state_seq, duration, env, scenario, path, bts_id, fps=2, figsize=(5.12, 5.12), basemap_resolution=1080, units_resolution=100, with_gif=True, verbose=True):
    """
    path must be of format "path/to/folder/file_name" without a . as it will save as .gif and .mp4
    """
    basemap_image = get_basemap_array(scenario, bts_id, basemap_resolution, figsize)
    compute_image = plot_step_factory(scenario, state_seq, env, units_resolution, bts_id)
    frames = []
    
    for t in tqdm(range(duration)) if verbose else range(duration): 
        fig, ax = plt.subplots(figsize=figsize)
        plt.imshow(np.rot90(basemap_image), extent=(0, basemap_image.shape[0], 0, basemap_image.shape[1]))
        image = compute_image(t)
        plt.imshow(np.rot90(image), extent=(0, basemap_image.shape[0], 0, basemap_image.shape[1]))
        plt.axis("off")
        plt.tight_layout(pad=0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        frames.append(Image.open(buf))
    # gif 
    if with_gif:
        imageio.mimsave(path+ ".gif", frames, duration=1000/fps, loop=0)
    # video
    video_path = path+ ".mp4"
    with imageio.get_writer(video_path, fps=fps, codec="libx264") as writer:  # You can adjust fps
        for frame in frames:
            writer.append_data(np.array(frame))  # Convert frame to numpy array and write to video

def plot_step_factory(scenario, state_seq, env, resolution, bts_id=None):
    allies_colors = jnp.array([[0, 0, 128, 255], [65, 105, 225, 255], [135, 206, 235, 255], [0, 255, 127, 255], [34, 139, 34, 255]], dtype=jnp.uint8)
    allies_trees_colors = jnp.array([[0, 72, 88, 255], [65, 143, 167, 255], [135, 216, 205, 255], [60, 189, 113, 255], [1, 68, 33, 255]], dtype=jnp.uint8)
    allies_colors = allies_colors[scenario.unit_types[bts_id, :env.num_allies]]
    allies_trees_colors = allies_trees_colors[scenario.unit_types[bts_id, :env.num_allies]]
    enemies_colors = jnp.array([[178, 34, 34, 255], [220, 20, 60, 255], [255, 99, 71, 255], [255, 192, 203, 255], [238, 130, 238, 255]], dtype=jnp.uint8)
    enemies_trees_colors = jnp.array([[148, 64, 34, 255], [180, 50, 50, 255], [225, 129, 81, 255], [225, 222, 203, 255], [142, 58, 89, 255]], dtype=jnp.uint8)
    enemies_colors = enemies_colors[scenario.unit_types[bts_id, env.num_allies:]]
    enemies_trees_colors = enemies_trees_colors[scenario.unit_types[bts_id, env.num_allies:]]
    mixed_team = jnp.array([[190, 0, 190, 255]])
    mixed_team_trees = jnp.array([[150, 40, 160, 255]])
    colors = jnp.concatenate([allies_colors, enemies_colors])
    colors_trees = jnp.concatenate([allies_trees_colors, enemies_trees_colors])
    scale = resolution/env.size
    units_image = jnp.zeros((resolution, resolution, 4), dtype=jnp.int32)
    state_idx = (bts_id,)
    def plot_step(t):
        idx = (t,) + state_idx
        indices = jnp.array(state_seq.unit_positions[idx] * scale, dtype=jnp.uint32)
        in_forest_indices = jnp.array(state_seq.unit_positions[idx], dtype=jnp.uint32)
        in_forest = scenario.terrain.forest[bts_id, in_forest_indices[:,0], in_forest_indices[:, 1]].reshape(env.num_agents, 1)   
        units_alive = (state_seq.unit_health[idx] > 0).reshape(env.num_agents, 1)   
        new_value = jnp.where(units_alive, jnp.where(in_forest, colors_trees, colors), units_image[indices[:, 0], indices[:, 1]])
        image = jax.lax.fori_loop(0, env.num_agents,  lambda i, base: base.at[indices[i,0], indices[i,1]].set(new_value[i]), units_image)
        allies_pos = jnp.zeros(units_image.shape, dtype=jnp.bool_).at[indices[:env.num_allies, 0], indices[:env.num_allies, 1]].set(jnp.where(units_alive[:env.num_allies], True, False))
        enemies_pos = jnp.zeros(units_image.shape, dtype=jnp.bool_).at[indices[env.num_allies:, 0], indices[env.num_allies:, 1]].set(jnp.where(units_alive[env.num_allies:], True, False))
        image = jnp.where(jnp.logical_and(allies_pos, enemies_pos), mixed_team, image)
        return image
    return jit(plot_step)


# %% [markdown]
# # Metrics 

# %%
@dataclass
class Metric:
    ally_health: chex.Array
    enemy_health: chex.Array
    duration: chex.Array
    coverage: chex.Array
    embedding: chex.Array
    actions_id: chex.Array
    
def compute_coverage_factory(env, unit_presence_size=10):
    res = int(env.size / unit_presence_size)
    def compute_coverage(init_state, state_seq, n_steps, duration):
        positions = vmap( lambda t: jnp.where((t <= duration)[:, jnp.newaxis, jnp.newaxis], state_seq.unit_positions[t, :, :env.num_allies], init_state.unit_positions[:, :env.num_allies]))(jnp.arange(n_steps))
        positions = jnp.concatenate(positions, axis=1)
        positions /= jnp.array([env.size, env.size])
        positions = jnp.clip(jnp.array(positions * jnp.array([res, res]), dtype=int), a_min=jnp.array([0,0]), a_max=jnp.array([res-1, res-1]))
        M = jnp.zeros((res,res))
        coverage = vmap(lambda pos: jnp.sum(M.at[pos[:, 0], pos[:, 1]].set(True))/(res*res))(positions)
        return coverage
    return compute_coverage

def compute_metric_factory(env, n_steps, n_bts, indexes):
    compute_coverage = compute_coverage_factory(env, unit_presence_size=env.unit_type_sight_ranges[0])
    bts_ids = jnp.arange(n_bts)
    
    def compute_metric(env, n_steps, state_seq, action_id_seq, scenario, init_state):  # calculate metrics suppose multiple scenarios and bts 
        allies_alive = jnp.sum(state_seq.unit_health[:, :, :env.num_allies]>0, axis=2)
        enemies_alive = jnp.sum(state_seq.unit_health[:, :, env.num_allies:]>0, axis=2)
        allies_end = jnp.where(jnp.all(allies_alive, axis=0), n_steps, jnp.argmax(allies_alive==0, axis=0))
        enemies_end = jnp.where(jnp.all(enemies_alive, axis=0), n_steps, jnp.argmax(enemies_alive==0, axis=0))
        duration = jnp.min(jnp.array([allies_end, enemies_end]), axis=0)
        final_health = vmap(lambda bts_id: state_seq.unit_health[jnp.where(duration[bts_id]==n_steps, n_steps-1, duration[bts_id]), bts_id] / env.unit_type_health[scenario.unit_types[bts_id]])(jnp.arange(n_bts)) 
        final_health = jnp.clip(final_health, 0, 1 )
        final_allies_health = jnp.mean(final_health[:, :env.num_allies], axis=1)
        final_enemies_health = jnp.mean(final_health[:, env.num_allies:], axis=1)
        frames = get_rbg_array(indexes, state_seq, env, scenario, bts_ids, units_resolution=100)
        E = vmap(vmap(fm.embed_img))(frames)  #img shape (H W C) and values in [0, 1].
        measures = Metric(final_allies_health, final_enemies_health, duration / n_steps, compute_coverage(init_state, state_seq, n_steps, duration), E, action_id_seq)
        return measures
    return lambda state_seq, actions_id_seq, init_state, scenario: compute_metric(env, n_steps, state_seq, actions_id_seq, scenario, init_state)


# %% [markdown]
# # GAME eval bts function

# %%
def eval_bi_bts_factory(config):
    n_agents = config["n_agents"]
    n_allies = config["n_allies"]
    n_enemies = n_agents-n_allies
    size = config["map_size"]
    n_steps = config["n_steps"]
    places = config['places']
    n_bts = config["n_bts"]
    units_resolution = int(config["units_resolution_scale"] * size)
    indexes = config["frames_indexes"]
    bt_max_size = config["bt_max_size"]
    frames_indexes = config["frames_indexes"]
    max_map_targets = config["max_map_targets"]
    
    # env 
    unit_type_sight_ranges = jnp.array([15, 15, 15, 15, 15, 10])
    env = Env(
        size = size,
        num_agents = n_agents,
        num_allies = n_allies,
        num_enemies = n_enemies,
        unit_type_radiuses = jnp.array([0.2, 0.1, 0.15, 0.1, 0.15, 0.1])*3,
        unit_type_health = jnp.array([24, 2, 12, 2, 2, 1]),
        unit_type_attacks = jnp.array([1, 3, 1, -2, 1, 0]),
        unit_type_attack_ranges = jnp.array([1, 15, 1, 1, 10, 0]),
        unit_type_sight_ranges = unit_type_sight_ranges,
        unit_type_velocities = jnp.array([1, 2, 6, 1, 1, 1]),  # distance per steps 
        unit_type_weapon_cooldowns = jnp.array([1, 1, 1, 1, 3, 0]),  # in number of steps 
        line_of_sight = compute_line_of_sight_discretization(unit_type_sight_ranges),
        grenade_radius = 5.,
    )
    ## initial scenario
    init_scenario = make_init_scenario(env, max_map_targets, n_bts, places)
    valid_sectors_fn = jit(valid_sectors_factory())
    valid_sectors = jnp.where(init_scenario.terrain.building + init_scenario.terrain.water>0, 1, 0)
    compute_distance_map = compute_distance_map_factory(env)  # already jitted 
    # init state 
    rng = random.PRNGKey(0)
    rng, init_rng = random.split(rng)
    init_state_fn = jit(init_state_factory(init_rng, env, n_bts))
    # step function 
    rng, key = random.split(rng)
    steps_rng = random.split(key, (n_steps, 2, n_agents))  # the 2 if for splitting in action and env step
    steps_rng = repeat(steps_rng, "steps two agents keys ->  steps two y agents keys", y=n_bts)  # copy rng for the bts 
    step = jit(step_factory(env, steps_rng, n_steps, all_variants, bt_max_size))
    metric_fn = jit(compute_metric_factory(env, n_steps, n_bts, frames_indexes))
    # step function
    def eval_bi_bts(a_bts, e_bts, unit_starting_sectors, unit_types, with_video=False, save_path=None):
        ## BT 
        a_predecessors, a_parents, a_passing_nodes, a_variant_ids = [], [], [], []
        e_predecessors, e_parents, e_passing_nodes, e_variant_ids = [], [], [], []
        for (a_bt, e_bt) in zip(a_bts, e_bts):
            a_predecessor, a_parent, a_passing_node, a_variant_id = txt2array(a_bt, bt_max_size)
            e_predecessor, e_parent, e_passing_node, e_variant_id = txt2array(e_bt, bt_max_size)
            # copy the same bt for all agents and scenarios
            a_predecessors.append(repeat(a_predecessor, "n_variants -> x n_variants", x=n_allies)) 
            a_parents.append(repeat(a_parent, "n_variants -> x n_variants", x=n_allies)) 
            a_passing_nodes.append(repeat(a_passing_node, "n_variants -> x n_variants", x=n_allies)) 
            a_variant_ids.append(repeat(a_variant_id, "n_variants -> x n_variants", x=n_allies))  
            e_predecessors.append(repeat(e_predecessor, "n_variants -> x n_variants", x=n_enemies)) 
            e_parents.append(repeat(e_parent, "n_variants -> x n_variants", x=n_enemies)) 
            e_passing_nodes.append(repeat(e_passing_node, "n_variants -> x n_variants", x=n_enemies))  
            e_variant_ids.append(repeat(e_variant_id, "n_variants -> x n_variants", x=n_enemies))
        predecessors = jnp.concatenate((jnp.array(a_predecessors), jnp.array(e_predecessors)), axis=1)
        parents = jnp.concatenate((jnp.array(a_parents), jnp.array(e_parents)), axis=1)
        passing_nodes = jnp.concatenate((jnp.array(a_passing_nodes), jnp.array(e_passing_nodes)), axis=1)
        variant_ids = jnp.concatenate((jnp.array(a_variant_ids), jnp.array(e_variant_ids)), axis=1)
        ## scenario
        # starting sectors 
        unit_starting_sectors = vmap(valid_sectors_fn)(jnp.array(unit_starting_sectors), valid_sectors)
        scenario = replace(init_scenario, unit_starting_sectors=unit_starting_sectors)
        # unit types 
        scenario = replace(scenario, unit_types=unit_types)
        ## running 
        init_state = init_state_fn(scenario)
        carry = (scenario, init_state, predecessors, parents, passing_nodes, variant_ids)
        carry, seq = jax.lax.scan(step, carry, jnp.arange(n_steps))
        scenario = carry[0]
        state_seq, action_seq, action_id_seq = seq
        measures = metric_fn(state_seq, action_id_seq, init_state, scenario)
        if with_video and save_path is not None:
            for bts_id in range(n_bts):
                path = save_path + f"_{bts_id}"
                plot_episode_jax(state_seq, n_steps, env, scenario, path, bts_id=bts_id, fps=24, figsize=(5.12, 5.12), basemap_resolution=1080, units_resolution=size, with_gif=False, verbose=False)
        return measures 
    return eval_bi_bts


# %% [markdown]
# # configs

# %%
if __name__ == "__main__":
    print("main")

    # %%
    n_agents = 10 * 64
    n_bts = 2
    config = {
        "places": ["blank"]*n_bts,
        "bt_max_size": 100,
        "map_size": 100,
        "n_agents": n_agents,
        "n_allies": n_agents//2,
        "n_steps": 100,
        "frames_indexes": jnp.array([0, 24, 49, 74, 99]),
        "units_resolution_scale": 1,
        "seed": 0,
        "n_bts": n_bts,
        "max_map_targets": 4,
    }

# %% [raw]
#     eval_fn = eval_bi_bts_factory(config)

# %% [raw]
#     a_bts = [bt, "A (stand)"]
#     e_bts = [bt2, "A (stand)"]
#     n_bts = len(a_bts)
#     num_agents = n_agents
#     num_allies = n_agents//2
#     num_enemies = n_agents - n_agents//2
#     size = 100
#     unit_starting_sectors = jnp.array([[[0,0,1,1]] * num_agents, [[0.4,0.4,0.2,0.2]] *num_agents ])
#     unit_types = jnp.array([[0]*num_agents, [1]*num_agents], dtype=jnp.uint8)
#     unit_target_position_id = jnp.array([[0]*num_allies + [1]*num_enemies] * n_bts, dtype=jnp.int32)
#     target_positions = jnp.array([[[3*size/4, size/4]]*num_agents, [[size/4, 3*size/4]]*num_agents], dtype=jnp.int32)
#
#     with_video = True
#     save_path = utils.ROOT_PATH + f"figures/jax_{0}"
#     measures = eval_fn(a_bts, e_bts, unit_starting_sectors, unit_types, with_video, save_path)

    # %%
    n_agents = config["n_agents"]
    n_allies = config["n_allies"]
    n_enemies = n_agents-n_allies
    size = config["map_size"]
    n_steps = config["n_steps"]
    places = config['places']
    n_bts = config['n_bts']
    units_resolution = int(config["units_resolution_scale"] * size)
    indexes = config["frames_indexes"]
    bt_max_size = config["bt_max_size"]
    frames_indexes = config["frames_indexes"]
    max_map_targets = config["max_map_targets"]

# %% [markdown]
# ## env

    # %%
    unit_type_sight_ranges = jnp.array([15, 15, 15, 15, 15, 10])
    env = Env(
        size = size,
        num_agents = n_agents,
        num_allies = n_allies,
        num_enemies = n_enemies,
        unit_type_radiuses = jnp.array([0.2, 0.1, 0.15, 0.1, 0.15, 0.1])*3,
        unit_type_health = jnp.array([24, 2, 12, 2, 2, 1]),
        unit_type_attacks = jnp.array([1, 3, 1, -2, 1, 0]),
        unit_type_attack_ranges = jnp.array([1, 15, 1, 1, 10, 0]),
        unit_type_sight_ranges = unit_type_sight_ranges,
        unit_type_velocities = jnp.array([1, 2, 6, 1, 1, 1]),  # distance per steps 
        unit_type_weapon_cooldowns = jnp.array([1, 1, 1, 1, 3, 0]),  # in number of steps 
        line_of_sight = compute_line_of_sight_discretization(unit_type_sight_ranges),
        grenade_radius = 5.,
    )

# %% [markdown]
# ## init scenario

    # %%
    scenario = make_init_scenario(env, max_map_targets, n_bts, places)

# %% [markdown]
# ## init state function

    # %%
    rng = random.PRNGKey(0)
    rng, init_rng = random.split(rng)
    init_state_fn = jit(init_state_factory(init_rng, env, n_bts))

# %% [markdown]
# ## step function

    # %%
    rng, key = random.split(rng)
    steps_rng = random.split(key, (n_steps, 2, n_agents))  # the 2 if for splitting in action and env step
    steps_rng = repeat(steps_rng, "steps two agents keys ->  steps two y agents keys", y=n_bts)  # copy rng for the bts 
    step = jit(step_factory(env, steps_rng, n_steps, all_variants, bt_max_size))
    metric_fn = compute_metric_factory(env, n_steps, n_bts, frames_indexes)

# %% [markdown]
# ## update scenario

# %% [markdown]
# ### initial_positions

    # %%
    valid_sectors_fn = jit(valid_sectors_factory())
    valid_sectors = jnp.where(scenario.terrain.building + scenario.terrain.water>0, 1, 0)

    unit_starting_sectors = jnp.array([
        [[0.33, 0.1, 0.33, 0.1]] * env.num_allies + [[0.33, 0.9-0.1, 0.33, 0.1]] * env.num_enemies, 
        [[0.33, 0.1, 0.33, 0.1]] * env.num_allies + [[0.33, 0.9-0.1, 0.33, 0.1]] * env.num_enemies, 
    ])
    unit_starting_sectors = vmap(valid_sectors_fn)(jnp.array(unit_starting_sectors), valid_sectors)
    scenario = replace(scenario, unit_starting_sectors=unit_starting_sectors)

# %% [markdown]
# ### targets positions

# %% [raw]
#     compute_distance_map = compute_distance_map_factory(env)  # already jitted 
#
#     unit_target_position_id = jnp.array([
#         [0] * env.num_agents,
#         [0] * env.num_agents,
#     ], dtype=jnp.int32)
#
#     targets = jnp.array([
#         [[0.5*size, 0.75*size]] * env.num_agents,
#         
#         [[0.5*size, 0.75*size]] * env.num_agents,
#     ], dtype=jnp.int32)
#
#     for i in range(n_bts):
#         for j in range(max_map_targets):
#             scenario = replace(scenario, distance_map=scenario.distance_map.at[i, j].set(compute_distance_map(scenario.terrain[i], targets[i,j])))
#     scenario = replace(scenario, unit_target_position_id=unit_target_position_id)

# %% [markdown]
# ### unit_types 

    # %%
    unit_types = jnp.array([
        [1]*env.num_agents,
        [1]*env.num_agents,
    ], dtype=jnp.uint8)
    scenario = replace(scenario, unit_types=unit_types)

# %% [markdown]
# ## BTs

    # %%
    bt = """F(
        S( C (is_set_target A) :: F(A (attack random any) :: A (go_to A 25%)))::
        S(C (in_sight foe any) :: A (set_target A closest foe any))::
        A (move away_from closest friend any)
    )
    """
    
    bt2 = """F(
        A (go_to A 100%)::
        A (set_target A closest friend any)
    )
    """
    
    a_bts = [
       bt, 
       bt2,
    ]
    e_bts = ["A (stand)", "A (stand)"]

    # %%
    a_predecessors, a_parents, a_passing_nodes, a_variant_ids = [], [], [], []
    e_predecessors, e_parents, e_passing_nodes, e_variant_ids = [], [], [], []

    for (a_bt, e_bt) in zip(a_bts, e_bts):
        a_predecessor, a_parent, a_passing_node, a_variant_id = txt2array(a_bt, bt_max_size)
        e_predecessor, e_parent, e_passing_node, e_variant_id = txt2array(e_bt, bt_max_size)
        # copy the same bt for all agents and scenarios
        a_predecessors.append(repeat(a_predecessor, "n_variants -> x n_variants", x=n_allies)) 
        a_parents.append(repeat(a_parent, "n_variants -> x n_variants", x=n_allies)) 
        a_passing_nodes.append(repeat(a_passing_node, "n_variants -> x n_variants", x=n_allies)) 
        a_variant_ids.append(repeat(a_variant_id, "n_variants -> x n_variants", x=n_allies))  
        e_predecessors.append(repeat(e_predecessor, "n_variants -> x n_variants", x=n_enemies)) 
        e_parents.append(repeat(e_parent, "n_variants -> x n_variants", x=n_enemies)) 
        e_passing_nodes.append(repeat(e_passing_node, "n_variants -> x n_variants", x=n_enemies))  
        e_variant_ids.append(repeat(e_variant_id, "n_variants -> x n_variants", x=n_enemies))

    predecessors = jnp.concatenate((jnp.array(a_predecessors), jnp.array(e_predecessors)), axis=1)
    parents = jnp.concatenate((jnp.array(a_parents), jnp.array(e_parents)), axis=1)
    passing_nodes = jnp.concatenate((jnp.array(a_passing_nodes), jnp.array(e_passing_nodes)), axis=1)
    variant_ids = jnp.concatenate((jnp.array(a_variant_ids), jnp.array(e_variant_ids)), axis=1)

# %% [markdown]
# ## run step function

    # %%
    init_state = init_state_fn(scenario)

    # %%
    carry = (scenario, init_state, predecessors, parents, passing_nodes, variant_ids)
    carry, seq = jax.lax.scan(step, carry, jnp.arange(n_steps))
    scenario = carry[0]
    state_seq, action_seq, action_id_seq = seq

# %% [markdown]
# ## metrics

    # %%
    measures = metric_fn(state_seq, action_id_seq, init_state, scenario)

    # %%
    measures.duration

# %% [markdown]
# ## Plot

    # %%
    for bts_id in range(n_bts):
        path = utils.ROOT_PATH + f"figures/jax_{bts_id}"
        plot_episode_jax(state_seq, n_steps, env, scenario, path, bts_id=bts_id, fps=24, figsize=(5.12, 5.12), basemap_resolution=1080, units_resolution=size)

# %%

# %%
