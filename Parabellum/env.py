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
import sys
sys.path.append("..")  # import utils from parent folder 

# %%
from functools import partial
import jax
import jax.numpy as jnp
from jax import random, vmap, Array, jit, tree_util
from jax.ops import segment_sum
from flax.struct import dataclass
import chex
import io
from PIL import Image
import imageio
import utils 
from dataclasses import replace
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm
from einops import repeat
from jax.scipy.signal import convolve
import tps 
from atomics import STAND, NONE, MOVE, ATTACK, AREA_ATTACK, HEAL, SET_TARGETS, Action, NONE_ACTION, STAND_ACTION

# %%
from time import time
from jax_tqdm import scan_tqdm  # require to update jax, jaxmarl, chex, flax 


# %% [markdown]
# # Dataclasses

# %%
@dataclass 
class State:
    unit_positions: chex.Array
    unit_health: chex.Array 
    unit_cooldowns: chex.Array
    unit_in_sight_distance: chex.Array
    time: chex.Array
    
    def __getitem__(self, index):  # to allow slicing operations
        return State(
            unit_positions=self.unit_positions[index],
            unit_health=self.unit_health[index],
            unit_cooldowns=self.unit_cooldowns[index],
            unit_in_sight_distance=self.unit_in_sight_distance[index],
            time=self.time[index],
        )
    
@dataclass
class Scenario:
    unit_types: chex.Array
    unit_team: chex.Array
    unit_target_position_id: chex.Array  # the idx of the distance_map (used by the following map atomics)
    unit_starting_sectors: chex.Array  # the inital spawing aera for each unit
    terrain: tps.Terrain
    distance_map: chex.Array  # the distance map used by the following map atomics (arbitrary number of them)
    movement_randomness: chex.Array
    units_push_back_firmness: chex.Array
    targets_position: chex.Array  # position of targets for the go_to atomics (!= follow_map) 
    
    def __getitem__(self, index):  # to allow slicing operations
        return Scenario(
            unit_types=self.unit_types[index],
            unit_team=self.unit_team[index],
            unit_target_position_id=self.unit_target_position_id[index],
            terrain=self.terrain[index],
            unit_starting_sectors=self.unit_starting_sectors[index],
            distance_map=self.distance_map[index],
            movement_randomness=self.movement_randomness[index],
            units_push_back_firmness=self.units_push_back_firmness[index],
            targets_position=self.targets_position[index],
        )

    def set_distance_map(self, distance_map, unit_target_position_id):
        return Scenario(
            unit_types=self.unit_types,
            unit_team=self.unit_team,
            unit_target_position_id=unit_target_position_id,
            terrain=self.terrain,
            unit_starting_sectors=self.unit_starting_sectors,
            distance_map=distance_map,
            movement_randomness=self.movement_randomness,
            units_push_back_firmness=self.units_push_back_firmness,
            targets_position=self.targets_position,
        )

    
@dataclass 
class Env:
    size: int
    num_agents: int 
    num_allies: int
    num_enemies: int
    unit_type_radiuses: chex.Array
    unit_type_health: chex.Array
    unit_type_attacks: chex.Array
    unit_type_attack_ranges: chex.Array
    unit_type_sight_ranges: chex.Array
    unit_type_velocities: chex.Array  # distance per steps 
    unit_type_weapon_cooldowns: chex.Array  # in number of steps 
    line_of_sight: chex.Array
    grenade_radius: float 
#    unit_type_names: list  # Jax didn't like the string list in the tree_fn of btc2sim.bt


# %% [markdown]
# # Functions 

# %%
def stack_state(init_state, state_seq):
    stacked_unit_positions = jnp.stack([init_state.unit_positions] + [state_seq.unit_positions[i] for i in range(len(state_seq.unit_positions))])
    stacked_unit_health = jnp.stack([init_state.unit_health] + [state_seq.unit_health[i] for i in range(len(state_seq.unit_health))])
    stacked_unit_cooldowns = jnp.stack([init_state.unit_cooldowns] + [state_seq.unit_cooldowns[i] for i in range(len(state_seq.unit_cooldowns))])
    stacked_time = jnp.stack([init_state.time] + [state_seq.time[i] for i in range(len(state_seq.time))])
    stacked_unit_in_sight_distance = jnp.stack([init_state.unit_in_sight_distance] + [state_seq.unit_in_sight_distance[i] for i in range(len(state_seq.unit_in_sight_distance))])
    return State(
        unit_positions=stacked_unit_positions,
        unit_health=stacked_unit_health,
        unit_cooldowns=stacked_unit_cooldowns,
        unit_in_sight_distance=stacked_unit_in_sight_distance,
        time=stacked_time,
    )


# %% [markdown]
# ## Scenario

# %%
def valid_sectors_factory():
    """
    sectors must be of shape (num_units, 4) where sectors[i] = (x, y, width, height) of the ith unit's spawning sector (in % of the real map)
    """
    def compute_valid_sector(sector, invalid_spawn_areas):
        width, height = invalid_spawn_areas.shape
        coordx, coordy = jnp.array(sector[0] * width, dtype=jnp.int32), jnp.array(sector[1] * height, dtype=jnp.int32)
        valid_area = 1-invalid_spawn_areas
        valid_area = jnp.where(jnp.arange(valid_area.shape[0]) >= coordx, valid_area.T, 0).T
        valid_area = jnp.where(jnp.arange(valid_area.shape[0]) <= coordx + jnp.ceil(sector[2] * width), valid_area.T, 0).T
        valid_area = jnp.where(jnp.arange(valid_area.shape[1]) >= coordy, valid_area, 0)
        valid_area = jnp.where(jnp.arange(valid_area.shape[1]) <= coordy + jnp.ceil(sector[3] * height), valid_area, 0)
        return valid_area
    return vmap(compute_valid_sector, in_axes=(0, None))


# %%
def stack_terrain(terrains):
    """ stack a list of Terrain dataclasses in one """
    stacked_building = jnp.stack([t.building for t in terrains], axis=0)
    stacked_water = jnp.stack([t.water for t in terrains], axis=0)
    stacked_forest = jnp.stack([t.forest for t in terrains], axis=0)
    stacked_basemap = jnp.stack([t.basemap for t in terrains], axis=0)
    return tps.Terrain(building=stacked_building, water=stacked_water, forest=stacked_forest, basemap=stacked_basemap)


# %% [markdown]
# ## Distance map

# %%
def compute_distance_map_factory(env):
    """
    returns a jited function to compute the distance_map for the scenario given a target position 
    """
    kernel = jnp.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
    def compute_distance_map(terrain, starting_pos):
        valid = ~jnp.logical_or(terrain.building, terrain.water)
        distance_map = jnp.ones(valid.shape, dtype=jnp.uint32) * env.size**2 
        current_cells = jnp.zeros(valid.shape, dtype=jnp.bool_)
        current_cells = current_cells.at[starting_pos[0], starting_pos[1]].set(True)

        def init_cond_fn(carry):
            _, current_cells, i = carry
            return jnp.all(jnp.where(current_cells, jnp.logical_and(current_cells, ~valid), True))  # while all current are not valid (the target was invalid)

        def init_body_fn(carry):
            distance_map, current_cells, i = carry
            distance_map = jnp.where(current_cells, jnp.array([i], dtype=jnp.uint32), distance_map)
            current_cells = jnp.logical_xor(convolve(current_cells, kernel, mode='same'), current_cells)  # compute neighbors
            current_cells = jnp.logical_and(current_cells, distance_map == env.size**2)  # remove already visited
            return (distance_map, current_cells, i+1)
        
        def main_cond_fn(carry):
            _, current_cells, i = carry
            return jnp.logical_and(i < env.size**2, jnp.any(current_cells))
        
        def main_body_fn(carry):
            distance_map, current_cells, i = carry
            distance_map = jnp.where(current_cells, jnp.array([i], dtype=jnp.uint32), distance_map)
            current_cells = jnp.logical_xor(convolve(current_cells, kernel, mode='same'), current_cells)  # compute neighbors
            current_cells = jnp.logical_and(current_cells, distance_map == env.size**2)  # remove already visited
            current_cells = jnp.logical_and(current_cells, valid)  # remove invalid 
            return (distance_map, current_cells, i+1)

        init_val = (distance_map, current_cells, 0) 
        _, current_cells, i = jax.lax.while_loop(init_cond_fn, init_body_fn, init_val)
        init_val = (distance_map, jnp.logical_and(current_cells, valid), i) 
        return jax.lax.while_loop(main_cond_fn, main_body_fn, init_val)[0]
    
    return jit(compute_distance_map)


# %% [markdown]
# # Env

# %%
def has_line_of_sight(obstacles, source, target, env):  
    # suppose that the target position is in sight_range of source, otherwise the line of sight might miss some cells
    current_line_of_sight = source[:, jnp.newaxis] * (1-env.line_of_sight) + env.line_of_sight * target[:, jnp.newaxis]
    cells = jnp.array(current_line_of_sight, dtype=jnp.int32)
    in_sight = obstacles[cells[0], cells[1]].sum() == 0
    return in_sight

def compute_unit_in_sight_distance(env, scenario, state):
    def aux(agent_id):
        dist_matrix = jnp.linalg.norm(state.unit_positions[agent_id]-state.unit_positions, axis=-1)
        dist_matrix = dist_matrix.at[agent_id].set(jnp.inf)
        dist_matrix = jnp.where(dist_matrix <= env.unit_type_sight_ranges[scenario.unit_types[agent_id]], dist_matrix, jnp.inf)  # in sight distance
        dist_matrix = jnp.where(state.unit_health > 0, dist_matrix, jnp.inf)  # alive units
        obstacles = (scenario.terrain.building + scenario.terrain.forest)  # cannot see through building and forest
        in_sight = vmap(has_line_of_sight, in_axes=(None, None, 0, None))(obstacles, state.unit_positions[agent_id], state.unit_positions, env)
        dist_matrix = jnp.where(in_sight, dist_matrix, jnp.inf)  # in line of sight (no obstacle)
        return dist_matrix
    return vmap(aux)(jnp.arange(env.num_agents))

def compute_line_of_sight_discretization(unit_type_sight_ranges):
    resolution = jnp.array(jnp.max(unit_type_sight_ranges), dtype=jnp.int32) * 2
    return jnp.tile(jnp.linspace(0, 1, resolution), (2, 1))  # the constant line of sight discretization 


# %%
def farthest_reachable_position(source, target, env, scenario):
    # suppose that the target is in sight_range of source, otherwise the line of sight might miss some cells
    obstacles = (scenario.terrain.building + scenario.terrain.water)
    current_line_of_sight = source[:, jnp.newaxis] * (1-env.line_of_sight) + env.line_of_sight * target[:, jnp.newaxis]
    cells = jnp.array(current_line_of_sight, dtype=jnp.int32)
    obstacle_mask = obstacles[cells[0], cells[1]] != 0
    is_out_of_map = jnp.logical_or(current_line_of_sight <= 0, current_line_of_sight >= env.size)
    is_out_of_map = jnp.logical_or(is_out_of_map[0], is_out_of_map[1])
    obstacle_mask = jnp.logical_or(obstacle_mask, is_out_of_map) # taking the map bounds in consideration
    farthest_visible_target = current_line_of_sight[:, jnp.argmax(obstacle_mask)-1]
    pos = jnp.where(obstacle_mask.sum() != 0, farthest_visible_target, target)
    return pos


# %%
def push_units_away(env, scenario, health, pos): 
    delta_matrix = pos[:, None] - pos[None, :]
    dist_matrix = jnp.linalg.norm(delta_matrix, axis=-1) + jnp.identity(env.num_agents) + 1e-6
    radius_vector = jnp.where(health > 0, env.unit_type_radiuses[scenario.unit_types], 0)  # ignore dead units 
    radius_matrix = radius_vector[:, None] + radius_vector[None, :]
    overlap_term = jax.nn.relu(radius_matrix / dist_matrix - 1.0)
    unit_positions = pos + scenario.units_push_back_firmness * jnp.sum(delta_matrix * overlap_term[:, :, None], axis=1) / 2
    return unit_positions


# %%
def unit_move(health, positions, action, rng):
    will_move = jnp.logical_and(action.kind == MOVE, health>0)
    return positions + jnp.where(will_move, action.value, jnp.zeros(2)) + random.normal(rng, (2,)) * 0.02

def unit_attack_and_heal(current_cooldown, health, action, attack_damage):
    target = jnp.array(action.value[0], dtype=jnp.uint32)
    will_attack = jnp.logical_and(jnp.logical_or(action.kind == ATTACK, action.kind == HEAL), jnp.logical_and(current_cooldown <= 0, health > 0))
    damages = jnp.where(will_attack, attack_damage, 0)  # suppose negative value for healing 
    return damages, target

def set_targets_per_team(scenario, action, rng):
    targets_position = scenario.targets_position
    for i, set_target in enumerate(SET_TARGETS):
        for team in [0, 1]:
            target_id = jnp.where(team == 1, 2*i, 2*i+1)  # even = allies targets | odd = enemies targets
            will_set_target = jnp.logical_and(action.kind == set_target, scenario.unit_team == team)
            random_values_to_pick = jnp.where(will_set_target, 0., jnp.inf) + random.uniform(rng, will_set_target.shape)*0.5
            action_id = jnp.argmin(random_values_to_pick)
            targets_position = targets_position.at[target_id].set(jnp.where(action.kind[action_id] == set_target, action.value[action_id], targets_position[target_id]))
    return replace(scenario, targets_position=targets_position)

def unit_cooldown(current_cooldown, health, action, reset_cooldown):
    use_cooldown = jnp.logical_and(jnp.logical_or(jnp.logical_or(action.kind == ATTACK, action.kind == HEAL), action.kind == AREA_ATTACK), jnp.logical_and(current_cooldown <= 0, health > 0))
    return jnp.where(use_cooldown, reset_cooldown, current_cooldown-1)

def unit_area_attack(env, current_cooldown, positions, health, action, attack_damage):
    target = jnp.array(action.value[0], dtype=jnp.uint32)
    will_attack = jnp.logical_and(action.kind == AREA_ATTACK, jnp.logical_and(current_cooldown <= 0, health > 0))
    all_units_damages = jnp.where(jnp.linalg.norm(positions-positions[target], axis=1) <= env.grenade_radius, jnp.where(will_attack, attack_damage, 0), 0)
    return all_units_damages

def env_step(env, scenario, state, action, rng):
    # apply attack (without checking the distances)
    scenario = set_targets_per_team(scenario, action, rng[0])  # should do something cleaner for the rng 
    damages, targets = vmap(unit_attack_and_heal)(state.unit_cooldowns, state.unit_health, action, env.unit_type_attacks[scenario.unit_types])  # vmap on the num_agents
    total_damage = segment_sum(damages, targets, num_segments=env.num_agents)  # doing a bit of magic to sum the damages for each unit considering the targets
    area_damage_damages = vmap(unit_area_attack, in_axes=(None, 0, None, 0, 0, 0))(env, state.unit_cooldowns, state.unit_positions, state.unit_health, action, env.unit_type_attacks[scenario.unit_types])
    total_damage = total_damage + jnp.sum(area_damage_damages, axis=0)
    state = replace(state, unit_health=jnp.clip(state.unit_health - total_damage, 0, env.unit_type_health[scenario.unit_types]))
    state = replace(state, unit_cooldowns=vmap(unit_cooldown)(state.unit_cooldowns, state.unit_health, action, env.unit_type_weapon_cooldowns[scenario.unit_types]))
    # apply move (without checking the distances)
    new_pos = vmap(unit_move)(state.unit_health, state.unit_positions, action, rng)  # vmap on the num_agents
    new_pos = vmap(farthest_reachable_position, in_axes=(0,0, None, None))(state.unit_positions, new_pos, env, scenario)  # need to check for collision
    final_pos = push_units_away(env, scenario, state.unit_health, new_pos)  # the units push each others 
    final_pos = vmap(farthest_reachable_position, in_axes=(0,0, None, None))(new_pos, final_pos, env, scenario)
    state = replace(state, unit_positions=final_pos)
    state = replace(state, time=state.time+1)
    state = replace(state, unit_in_sight_distance=compute_unit_in_sight_distance(env, scenario, state))
    return state, scenario
