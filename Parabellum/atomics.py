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
from jax import vmap, random
from jax.lax import fori_loop
import jax.numpy as jnp
from flax.struct import dataclass
import chex

# %%
from grammar import all_variants, unit_types, targets, actions, conditions
from BT_array import Parent 

# %% [markdown]
# # Atomics

# %%
target_types = {
    "spearmen": 0,
    "archer": 1,
    "cavalry": 2,
    "healer": 3,
    "grenadier": 4,
    "any": None,
}


# %%
@dataclass
class Status:  # for behavior tree
    SUCCESS: int = 1
    FAILURE: int = -1
    NONE: int = 0


# %% [markdown]
# ## Action class

# %%
NONE, STAND, MOVE, ATTACK, HEAL, AREA_ATTACK = jnp.array(-1), jnp.array(0), jnp.array(1), jnp.array(2), jnp.array(3), jnp.array(4)
SET_TARGETS = jnp.array([5+i for i in range(len(targets))])


# %%
@dataclass
class Action:
    kind: chex.Array
    value: chex.Array
    
    def __getitem__(self, index):  # to allow slicing operations
        return Action(
            kind=self.kind[index],
            value=self.value[index],
        )
        
    def set_item(self, index, new_value):
        # Perform an in-place update to kind and value at the specified index
        return Action( 
            kind = self.kind.at[index].set(new_value.kind),
            value = self.value.at[index].set(new_value.value)
        )

    def where(self, condition, false_value):
        return Action(
            kind = jnp.where(condition, self.kind, false_value.kind),
            value = jnp.where(condition, self.value, false_value.value),
        )
    
    @classmethod
    def from_shape(cls, shape, dtype=jnp.float32):
        # Create an instance with empty arrays of the specified shape
        return cls(
            kind=jnp.ones(shape, dtype=dtype)*NONE,
            value=jnp.zeros(shape+(2,), dtype=dtype)
        )

    def conditional_action(condition, action_if_true, action_if_false):
        return Action(
            kind=jnp.where(condition, action_if_true.kind, action_if_false.kind),
            value=jnp.where(condition, action_if_true.value, action_if_false.value)
        )


# %%
NONE_ACTION = Action(NONE, jnp.zeros((2,), dtype=jnp.float32))
STAND_ACTION = Action(STAND, jnp.zeros((2,), dtype=jnp.float32))


# %% [markdown]
# ## Miscellaneous

# %%
def has_line_of_sight(obstacles, source, target, env):  
    # suppose that the target position is in sight_range of source, otherwise the line of sight might miss some cells
    current_line_of_sight = source[:, jnp.newaxis] * (1-env.line_of_sight) + env.line_of_sight * target[:, jnp.newaxis]
    cells = jnp.array(current_line_of_sight, dtype=jnp.int32)
    in_sight = obstacles[cells[0], cells[1]].sum() == 0
    return in_sight


# %% [markdown]
# ## Action atomics

# %% [markdown]
# ### Stand

# %%
def stand_factory(all_variants):
    def stand(env, scenario, state, agent_id, variants_status, variants_action):
        variant_id = all_variants.index("stand")
        return variants_status.at[variant_id].set(Status.SUCCESS), variants_action.set_item(variant_id, STAND_ACTION)
    return stand


# %% [markdown]
# ### Debug

# %%
def debug_factory(all_variants):
    def debug(env, scenario, state, agent_id, variants_status, variants_action):
        def aux(variant_id, motion):
            return variants_status.at[variant_id].set(Status.SUCCESS),  variants_action.set_item(variant_id, Action(MOVE, motion))
        velocity = env.unit_type_velocities[scenario.unit_types[agent_id]]
        variants_status, variants_action = aux(all_variants.index("debug north"), jnp.array([0, velocity]))
        variants_status, variants_action = aux(all_variants.index("debug south"), jnp.array([0, -velocity]))
        variants_status, variants_action = aux(all_variants.index("debug east"), jnp.array([velocity, 0]))
        variants_status, variants_action = aux(all_variants.index("debug west"), jnp.array([-velocity, 0]))
        return variants_status, variants_action
    return debug


# %% [markdown]
# ### Attack

# %%
def attack_factory(all_variants):
    def attack(env, scenario, state, rng, agent_id, variants_status, variants_action):
        can_attack = jnp.logical_and(state.unit_cooldowns[agent_id] <= 0, scenario.unit_types[agent_id] != target_types["healer"])
        attack_type = jnp.where(scenario.unit_types[agent_id] == target_types["grenadier"], AREA_ATTACK, ATTACK)
        close_dist_matrix = state.unit_in_sight_distance[agent_id]
        close_dist_matrix = jnp.where(scenario.unit_team != scenario.unit_team[agent_id], close_dist_matrix, jnp.inf)  # only enemies
        close_dist_matrix = jnp.where(close_dist_matrix <= env.unit_type_attack_ranges[scenario.unit_types[agent_id]], close_dist_matrix, jnp.inf)  # in attack range 
        far_dist_matrix = jnp.where(close_dist_matrix == jnp.inf, -jnp.inf, close_dist_matrix)  # we want to use argmax
        def aux(value, extremum, variant, variants_status, variants_action):
            target_id = (jnp.argmax if extremum == "max" else jnp.argmin)(value)
            variant_id = all_variants.index("attack " + variant)
            flag = jnp.where(value[target_id] != (jnp.inf if extremum == "min" else -jnp.inf), Status.SUCCESS, Status.FAILURE)
            flag = jnp.where(can_attack, flag, Status.FAILURE)
            action = Action(kind=jnp.where(flag==Status.SUCCESS, attack_type, NONE), value=jnp.where(flag ==Status.SUCCESS, jnp.array([target_id, 0], dtype=jnp.float32), jnp.zeros(2)))
            variants_status = variants_status.at[variant_id].set(flag)
            variants_action = variants_action.set_item(variant_id, action)
            return variants_status, variants_action
        
        # random 
        value = jnp.where(close_dist_matrix != jnp.inf, 1, -jnp.inf)
        value += random.uniform(rng, value.shape)*0.5
        variants_status, variants_action = aux(value, "max", "random any", variants_status, variants_action)
        # distances 
        variants_status, variants_action = aux(close_dist_matrix, "min", "closest any", variants_status, variants_action)
        variants_status, variants_action = aux(far_dist_matrix, "max", "farthest any", variants_status, variants_action)
        # health 
        min_health = jnp.where(close_dist_matrix != jnp.inf, state.unit_health, jnp.inf)
        min_health += random.uniform(rng, min_health.shape)*0.5
        max_health = jnp.where(min_health == jnp.inf, -jnp.inf, min_health)
        variants_status, variants_action = aux(min_health, "min", "weakest any", variants_status, variants_action)
        variants_status, variants_action = aux(max_health, "max", "strongest any", variants_status, variants_action)
        
        for unit_type in unit_types:
            units = scenario.unit_types == target_types[unit_type]     
            # random 
            variants_status, variants_action = aux(jnp.where(units, value, -jnp.inf), "max", f"random {unit_type}", variants_status, variants_action)
            # distances 
            variants_status, variants_action = aux(jnp.where(units, close_dist_matrix, jnp.inf), "min", f"closest {unit_type}", variants_status, variants_action)
            variants_status, variants_action = aux(jnp.where(units, far_dist_matrix, -jnp.inf), "max", f"farthest {unit_type}", variants_status, variants_action)     
            # health 
            variants_status, variants_action = aux(jnp.where(units, min_health, jnp.inf), "min", f"weakest {unit_type}", variants_status, variants_action)
            variants_status, variants_action = aux(jnp.where(units, max_health, -jnp.inf), "max", f"strongest {unit_type}", variants_status, variants_action)
       
        return variants_status, variants_action
    return attack 


# %% [markdown]
# ### Move 

# %%
def move_factory(all_variants):
    def move(env, scenario, state, rng, agent_id, variants_status, variants_action):
        close_dist_matrix = state.unit_in_sight_distance[agent_id]
        far_dist_matrix = jnp.where(close_dist_matrix == jnp.inf, -jnp.inf, close_dist_matrix)  # we want to use argmax
        foes = scenario.unit_team != scenario.unit_team[agent_id]
        friends = scenario.unit_team == scenario.unit_team[agent_id]
        
        def aux(value, extremum, variant, variants_status, variants_action):
            target_id = (jnp.argmax if extremum == "max" else jnp.argmin)(value)
            valid_target = value[target_id] != (jnp.inf if extremum == "min" else -jnp.inf)
            variant_id_toward, variant_id_away_from = all_variants.index("move toward " + variant), all_variants.index("move away_from "+variant)
            delta = state.unit_positions[target_id] - state.unit_positions[agent_id]
            norm = jnp.linalg.norm(delta)
            velocity = env.unit_type_velocities[scenario.unit_types[agent_id]]
            delta = jnp.where(norm<=velocity, delta, velocity*delta/norm)
            obstacles = (scenario.terrain.building + scenario.terrain.water)  # cannot cross building and water 
            
            can_move_toward_target = has_line_of_sight(obstacles, state.unit_positions[agent_id], state.unit_positions[agent_id]+delta, env)
            can_move_away_from_target = has_line_of_sight(obstacles, state.unit_positions[agent_id], state.unit_positions[agent_id]-delta, env)
            
            flag_toward = jnp.where(jnp.logical_and(can_move_toward_target, valid_target), Status.SUCCESS, Status.FAILURE)
            flag_away_from = jnp.where(jnp.logical_and(can_move_away_from_target, valid_target), Status.SUCCESS, Status.FAILURE)
            action_toward = Action(kind=jnp.where(flag_toward ==Status.SUCCESS, MOVE, NONE), value=jnp.where(flag_toward ==Status.SUCCESS, delta, jnp.zeros(2)))
            action_away_from = Action(kind=jnp.where(flag_away_from ==Status.SUCCESS, MOVE, NONE), value=jnp.where(flag_away_from ==Status.SUCCESS, -delta, jnp.zeros(2)))

            variants_status = variants_status.at[variant_id_toward].set(flag_toward) 
            variants_status = variants_status.at[variant_id_away_from].set(flag_away_from)
            variants_action = variants_action.set_item(variant_id_toward, action_toward)
            variants_action = variants_action.set_item(variant_id_away_from, action_away_from)
            return variants_status, variants_action

        # random 
        value = jnp.where(close_dist_matrix != jnp.inf, 1, -jnp.inf)
        value += random.uniform(rng, value.shape)*0.5
        #health 
        min_health = jnp.where(close_dist_matrix != jnp.inf, state.unit_health, jnp.inf)
        min_health += random.uniform(rng, min_health.shape)*0.5
        max_health = jnp.where(min_health == jnp.inf, -jnp.inf, min_health)
        
        for team_name, team in zip(["friend", "foe"], [friends, foes]):
            # random 
            variants_status, variants_action = aux(jnp.where(team, value, -jnp.inf), "max", f"random {team_name} any", variants_status, variants_action)
            # distance 
            variants_status, variants_action = aux(jnp.where(team, close_dist_matrix, jnp.inf), "min", f"closest {team_name} any", variants_status, variants_action)
            variants_status, variants_action = aux(jnp.where(team, far_dist_matrix, -jnp.inf), "max", f"farthest {team_name} any", variants_status, variants_action)
            # health 
            variants_status, variants_action = aux(jnp.where(team, min_health, jnp.inf), "min", f"weakest {team_name} any", variants_status, variants_action)
            variants_status, variants_action = aux(jnp.where(team, max_health, -jnp.inf), "max", f"strongest {team_name} any", variants_status, variants_action)
            for unit_type in unit_types:
                units = jnp.logical_and(scenario.unit_types == target_types[unit_type], team)
                # random 
                variants_status, variants_action = aux(jnp.where(team, value, -jnp.inf), "max", f"random {team_name} {unit_type}", variants_status, variants_action)
                # distance 
                variants_status, variants_action = aux(jnp.where(units, close_dist_matrix, jnp.inf), "min", f"closest {team_name} {unit_type}", variants_status, variants_action)
                variants_status, variants_action = aux(jnp.where(units, far_dist_matrix, -jnp.inf), "max", f"farthest {team_name} {unit_type}", variants_status, variants_action)
                # health 
                variants_status, variants_action = aux(jnp.where(units, min_health, jnp.inf), "min", f"weakest {team_name} {unit_type}", variants_status, variants_action)
                variants_status, variants_action = aux(jnp.where(units, max_health, -jnp.inf), "max", f"strongest {team_name} {unit_type}", variants_status, variants_action)
       
        return variants_status, variants_action
    return move


# %% [markdown]
# ### Follow map

# %%
def follow_map_factory(all_variants):
    n_direction = 8  # number of direction arround the unit (2pi/n_direction)
    n_step_size = 4  # number of steps in the direction up to the unit's velocity (should be at least equal to the max velocity so that it check every cells 

    def follow_map(env, scenario, state, rng, agent_id, variants_status, variants_action):
        candidates = jnp.array([[0,0]] + [ [step_size/n_step_size*jnp.cos(2*jnp.pi*theta/n_direction), step_size/n_step_size*jnp.sin(2*jnp.pi*theta/n_direction)] for theta in jnp.arange(n_direction) for step_size in jnp.arange(1, n_step_size+1)])
        candidates *= env.unit_type_velocities[scenario.unit_types[agent_id]]
        candidates_idx = jnp.array(state.unit_positions[agent_id] + candidates, dtype=jnp.uint32)
        candidates_idx = jnp.clip(candidates_idx, 0, env.size-1)

        distances = scenario.distance_map[scenario.unit_target_position_id[agent_id]][candidates_idx[:,0], candidates_idx[:,1]]
        distances += random.uniform(rng, distances.shape, minval=0.0, maxval=scenario.movement_randomness)  # to resolve tighs and give a more organic vibe 
        obstacles = (scenario.terrain.building + scenario.terrain.water)  # cannot walk through building and water
        in_sight = vmap(has_line_of_sight, in_axes=(None, None, 0, None))(obstacles, state.unit_positions[agent_id], state.unit_positions[agent_id] + candidates, env)
        distances_toward = jnp.where(in_sight, distances, env.size**2)  # in sight positions
        distances_away = jnp.where(distances_toward >= env.size**2, -1, distances_toward)
        
        for margin_name, margin in zip(["0%", "25%", "50%", "100%"], [0, 0.25, 0.5, 1.]):
            d = env.unit_type_sight_ranges[scenario.unit_types[agent_id]]
            for sense_name, target_idx in zip(["toward", "away_from"], [jnp.argmin(distances_toward), jnp.argmax(distances_away)]):
                if margin_name == "0%" and sense_name == "away_from":
                    d = jnp.inf
                else:
                    d *= margin 
                margin_check = distances_toward[0] > d if sense_name == "toward" else distances_away[0] < d
                need_to_move = target_idx != 0  # atomic return Failure if the agent doesn't need to move or can't move
                flag = jnp.where(jnp.logical_and(margin_check, need_to_move), Status.SUCCESS, Status.FAILURE) 
                action = Action(kind=jnp.where(flag==Status.SUCCESS, MOVE, NONE), value=candidates[target_idx])
                variant_id = all_variants.index(f"follow_map {sense_name} {margin_name}")
                variants_status, variants_action = variants_status.at[variant_id].set(flag), variants_action.set_item(variant_id, action)

        return variants_status, variants_action
    return follow_map


# %% [markdown]
# ### Heal

# %%
def heal_factory(all_variants):
    def heal(env, scenario, state, rng, agent_id, variants_status, variants_action):
        can_heal = jnp.logical_and(state.unit_cooldowns[agent_id] <= 0, scenario.unit_types[agent_id] == target_types["healer"])
        close_dist_matrix = state.unit_in_sight_distance[agent_id]
        close_dist_matrix = jnp.where(scenario.unit_team == scenario.unit_team[agent_id], close_dist_matrix, jnp.inf)  # only allies
        close_dist_matrix = jnp.where(close_dist_matrix <= env.unit_type_attack_ranges[scenario.unit_types[agent_id]], close_dist_matrix, jnp.inf)  # in attack range 
        far_dist_matrix = jnp.where(close_dist_matrix == jnp.inf, -jnp.inf, close_dist_matrix)  # we want to use argmax
        def aux(value, extremum, variant, variants_status, variants_action):
            target_id = (jnp.argmax if extremum == "max" else jnp.argmin)(value)
            variant_id = all_variants.index("heal " + variant)
            flag = jnp.where(value[target_id] != (jnp.inf if extremum == "min" else -jnp.inf), Status.SUCCESS, Status.FAILURE)
            flag = jnp.where(can_heal, flag, Status.FAILURE)
            action = Action(kind=jnp.where(flag==Status.SUCCESS, HEAL, NONE), value=jnp.where(flag==Status.SUCCESS, jnp.array([target_id, 0], dtype=jnp.float32), jnp.zeros(2)))
            variants_status = variants_status.at[variant_id].set(flag)
            variants_action = variants_action.set_item(variant_id, action)
            return variants_status, variants_action
        
        # random 
        value = jnp.where(close_dist_matrix != jnp.inf, 1, -jnp.inf)
        value += random.uniform(rng, value.shape)*0.5
        variants_status, variants_action = aux(value, "max", "random any", variants_status, variants_action)
        # distances 
        variants_status, variants_action = aux(close_dist_matrix, "min", "closest any", variants_status, variants_action)
        variants_status, variants_action = aux(far_dist_matrix, "max", "farthest any", variants_status, variants_action)
        # health 
        min_health = jnp.where(close_dist_matrix != jnp.inf, state.unit_health, jnp.inf)
        min_health += random.uniform(rng, min_health.shape)*0.5
        max_health = jnp.where(min_health == jnp.inf, -jnp.inf, min_health)
        variants_status, variants_action = aux(min_health, "min", "weakest any", variants_status, variants_action)
        variants_status, variants_action = aux(max_health, "max", "strongest any", variants_status, variants_action)
        
        for unit_type in unit_types:
            units = scenario.unit_types == target_types[unit_type]     
            # random 
            variants_status, variants_action = aux(jnp.where(units, value, -jnp.inf), "max", f"random {unit_type}", variants_status, variants_action)
            # distances 
            variants_status, variants_action = aux(jnp.where(units, close_dist_matrix, jnp.inf), "min", f"closest {unit_type}", variants_status, variants_action)
            variants_status, variants_action = aux(jnp.where(units, far_dist_matrix, -jnp.inf), "max", f"farthest {unit_type}", variants_status, variants_action)     
            # health 
            variants_status, variants_action = aux(jnp.where(units, min_health, jnp.inf), "min", f"weakest {unit_type}", variants_status, variants_action)
            variants_status, variants_action = aux(jnp.where(units, max_health, -jnp.inf), "max", f"strongest {unit_type}", variants_status, variants_action)
       
        return variants_status, variants_action
    return heal


# %% [markdown]
# ### Go_to

# %%
def go_to_factory(all_variants):
    def go_to(env, scenario, state, agent_id, variants_status, variants_action):
        for i, target in enumerate(targets):  # targets set in grammar 
            target_id = jnp.where(scenario.unit_team[agent_id]==1, 2*i, 2*i+1)  # even = allies targets | odd = enemies targets
            delta = scenario.targets_position[target_id] - state.unit_positions[agent_id]
            norm = jnp.linalg.norm(delta)
            velocity = env.unit_type_velocities[scenario.unit_types[agent_id]]
            delta = jnp.where(norm<=velocity, delta, velocity*delta/norm)
            for margin_name, margin in zip(["0%", "25%", "50%", "100%"], [0, 0.25, 0.5, 1.]):
                valid_target = jnp.all(scenario.targets_position[target_id] >= 0)
                target_out_of_margin = norm > (env.unit_type_sight_ranges[scenario.unit_types[agent_id]]*margin)
                flag = jnp.where(jnp.logical_and(valid_target, target_out_of_margin), Status.SUCCESS, Status.FAILURE)                
                action = Action(kind=jnp.where(flag==Status.SUCCESS, MOVE, NONE), value=jnp.where(flag==Status.SUCCESS, delta, jnp.zeros(2)))
                variant_id = all_variants.index(f"go_to {target} {margin_name}")
                variants_status = variants_status.at[variant_id].set(flag)
                variants_action = variants_action.set_item(variant_id, action)
        return variants_status, variants_action
    return go_to


# %% [markdown]
# ### Set_target 

# %%
def set_target_factory(all_variants):
    def set_target(env, scenario, state, rng, agent_id, variants_status, variants_action):
        close_dist_matrix = state.unit_in_sight_distance[agent_id]
        far_dist_matrix = jnp.where(close_dist_matrix == jnp.inf, -jnp.inf, close_dist_matrix)  # we want to use argmax
        foes = scenario.unit_team != scenario.unit_team[agent_id]
        friends = scenario.unit_team == scenario.unit_team[agent_id]
        
        def aux(value, extremum, variant, variants_status, variants_action):
            target_id = (jnp.argmax if extremum == "max" else jnp.argmin)(value)
            valid_target = value[target_id] != (jnp.inf if extremum == "min" else -jnp.inf)
            target_pos = state.unit_positions[target_id]
            for i, target in enumerate(targets):
                variant_id = all_variants.index(f"set_target {target} {variant}")
                flag= jnp.where(valid_target, Status.SUCCESS, Status.FAILURE)
                action = Action(kind=jnp.where(flag ==Status.SUCCESS, SET_TARGETS[i], NONE), value=jnp.where(flag == Status.SUCCESS, target_pos, scenario.targets_position[i]))
                variants_status = variants_status.at[variant_id].set(flag) 
                variants_action = variants_action.set_item(variant_id, action)
                return variants_status, variants_action

        # random 
        value = jnp.where(close_dist_matrix != jnp.inf, 1, -jnp.inf)
        value += random.uniform(rng, value.shape)*0.5
        #health 
        min_health = jnp.where(close_dist_matrix != jnp.inf, state.unit_health, jnp.inf)
        min_health += random.uniform(rng, min_health.shape)*0.5
        max_health = jnp.where(min_health == jnp.inf, -jnp.inf, min_health)
        
        for team_name, team in zip(["friend", "foe"], [friends, foes]):
            # random 
            variants_status, variants_action = aux(jnp.where(team, value, -jnp.inf), "max", f"random {team_name} any", variants_status, variants_action)
            # distance 
            variants_status, variants_action = aux(jnp.where(team, close_dist_matrix, jnp.inf), "min", f"closest {team_name} any", variants_status, variants_action)
            variants_status, variants_action = aux(jnp.where(team, far_dist_matrix, -jnp.inf), "max", f"farthest {team_name} any", variants_status, variants_action)
            # health 
            variants_status, variants_action = aux(jnp.where(team, min_health, jnp.inf), "min", f"weakest {team_name} any", variants_status, variants_action)
            variants_status, variants_action = aux(jnp.where(team, max_health, -jnp.inf), "max", f"strongest {team_name} any", variants_status, variants_action)
            for unit_type in unit_types:
                units = jnp.logical_and(scenario.unit_types == target_types[unit_type], team)
                # random 
                variants_status, variants_action = aux(jnp.where(team, value, -jnp.inf), "max", f"random {team_name} {unit_type}", variants_status, variants_action)
                # distance 
                variants_status, variants_action = aux(jnp.where(units, close_dist_matrix, jnp.inf), "min", f"closest {team_name} {unit_type}", variants_status, variants_action)
                variants_status, variants_action = aux(jnp.where(units, far_dist_matrix, -jnp.inf), "max", f"farthest {team_name} {unit_type}", variants_status, variants_action)
                # health 
                variants_status, variants_action = aux(jnp.where(units, min_health, jnp.inf), "min", f"weakest {team_name} {unit_type}", variants_status, variants_action)
                variants_status, variants_action = aux(jnp.where(units, max_health, -jnp.inf), "max", f"strongest {team_name} {unit_type}", variants_status, variants_action)
       
        return variants_status, variants_action
    return set_target


# %% [markdown]
# ## Conditions atomics

# %% [markdown]
# ### In Sight

# %% [raw]
# in_sight  : "in_sight" (foe | friend) (unit | any) 

# %%
def in_sight_factory(all_variants, n_agents):
    def in_sight(env, scenario, state, agent_id, variants_status):
        dist_matrix = state.unit_in_sight_distance[agent_id]
        foes = scenario.unit_team != scenario.unit_team[agent_id]
        friends = scenario.unit_team == scenario.unit_team[agent_id]
        any_unit = jnp.ones(n_agents, dtype=jnp.bool)
        def aux(target, unit_type):
            return jnp.where(jnp.any(jnp.where(jnp.logical_and(target, unit_type), dist_matrix, jnp.inf) != jnp.inf), Status.SUCCESS, Status.FAILURE)
        for team_name, team in zip(["friend", "foe"], [friends, foes]):
            variants_status = variants_status.at[all_variants.index(f"in_sight {team_name} any")].set(aux(team, any_unit))
            for unit_type in unit_types:
                units = scenario.unit_types == target_types[unit_type]
                variants_status = variants_status.at[all_variants.index(f"in_sight {team_name} {unit_type}")].set(aux(team, units))
        return variants_status
    return in_sight


# %% [markdown]
# ### In Reach

# %%
def in_reach_factory(all_variants, n_agents):
    def in_reach(env, scenario, state, agent_id, variants_status):
        foes = scenario.unit_team != scenario.unit_team[agent_id]
        friends = scenario.unit_team == scenario.unit_team[agent_id]
        any_unit = jnp.ones(n_agents, dtype=jnp.bool)
        dist_matrix = state.unit_in_sight_distance[agent_id]
        in_reach_0_from_me = dist_matrix <= env.unit_type_attack_ranges[scenario.unit_types[agent_id]]
        in_reach_1_from_me = dist_matrix <= (env.unit_type_attack_ranges[scenario.unit_types[agent_id]] + env.unit_type_velocities[scenario.unit_types[agent_id]])
        in_reach_2_from_me = dist_matrix <= (env.unit_type_attack_ranges[scenario.unit_types[agent_id]] + 2*env.unit_type_velocities[scenario.unit_types[agent_id]])
        in_reach_3_from_me = dist_matrix <= (env.unit_type_attack_ranges[scenario.unit_types[agent_id]] + 3*env.unit_type_velocities[scenario.unit_types[agent_id]])
        them_from_me = [in_reach_0_from_me, in_reach_1_from_me, in_reach_2_from_me, in_reach_3_from_me]
        in_reach_0_from_them = dist_matrix <= env.unit_type_attack_ranges[scenario.unit_types]
        in_reach_1_from_them = dist_matrix <= (env.unit_type_attack_ranges[scenario.unit_types] + env.unit_type_velocities[scenario.unit_types])
        in_reach_2_from_them = dist_matrix <= (env.unit_type_attack_ranges[scenario.unit_types] + 2*env.unit_type_velocities[scenario.unit_types])
        in_reach_3_from_them = dist_matrix <= (env.unit_type_attack_ranges[scenario.unit_types] + 3*env.unit_type_velocities[scenario.unit_types])
        me_from_them = [in_reach_0_from_them, in_reach_1_from_them, in_reach_2_from_them, in_reach_3_from_them]
        for source_name, sources in zip(["me_from_them", "them_from_me"], [me_from_them, them_from_me]):
            for type_name in unit_types + ["any"]:
                unit_type = any_unit if type_name == "any" else (scenario.unit_types == target_types[type_name])
                for team_name, unit_team in zip(["foe", "friend"], [foes, friends]):
                    for steps_name, in_reach_steps in zip(["0", "1", "2", "3"], sources):
                        variant_id = all_variants.index(f"in_reach {team_name} {source_name} {steps_name} {type_name}") 
                        variants_status = variants_status.at[variant_id].set(jnp.where(jnp.any(jnp.logical_and(jnp.logical_and(in_reach_steps, unit_type), unit_team)), Status.SUCCESS, Status.FAILURE))
        return variants_status
    return in_reach


# %% [markdown]
# ### Is_Type

# %%
def is_type_factory(all_variants):
    def is_type(env, scenario, state, agent_id, variants_status):
        for unit_type in unit_types:
            variants_status = variants_status.at[all_variants.index(f"is_type {unit_type}")].set(jnp.where(scenario.unit_types[agent_id] == target_types[unit_type], Status.SUCCESS, Status.FAILURE))
        return variants_status
    return is_type


# %% [markdown]
# ### Is Dying

# %%
def is_dying_factory(all_variants):
    def is_dying(env, scenario, state, agent_id, variants_status):
        dist_matrix = state.unit_in_sight_distance[agent_id]
        foes = jnp.where(scenario.unit_team != scenario.unit_team[agent_id], dist_matrix, jnp.inf)
        friends = jnp.where(scenario.unit_team == scenario.unit_team[agent_id], dist_matrix, jnp.inf)
        for threshold_name, threshold in zip(["25%", "50%", "75%"], [0.25, 0.5, 0.75]):
            flag = jnp.where(state.unit_health[agent_id]/env.unit_type_health[scenario.unit_types[agent_id]] <= threshold, Status.SUCCESS, Status.FAILURE)
            variants_status = variants_status.at[all_variants.index(f"is_dying self {threshold_name}")].set(flag)
            foes_flag = jnp.where(jnp.any(jnp.where(foes != jnp.inf, state.unit_health/env.unit_type_health[scenario.unit_types], 1) <= threshold), Status.SUCCESS, Status.FAILURE)
            variants_status = variants_status.at[all_variants.index(f"is_dying foe {threshold_name}")].set(foes_flag)
            friends_flag = jnp.where(jnp.any(jnp.where(friends != jnp.inf, state.unit_health/env.unit_type_health[scenario.unit_types], 1) <= threshold), Status.SUCCESS, Status.FAILURE)
            variants_status = variants_status.at[all_variants.index(f"is_dying friend {threshold_name}")].set(friends_flag)
        return variants_status
    return is_dying


# %% [markdown]
# ### Is in forest 

# %%
def is_in_forest_factory(all_variants):
    def is_in_forest(env, scenario, state, agent_id, variants_status):
        pos = state.unit_positions[agent_id].astype(jnp.uint32)
        variants_status = variants_status.at[all_variants.index(f"is_in_forest")].set(jnp.where(scenario.terrain.forest[pos[0], pos[1]], Status.SUCCESS, Status.FAILURE))
        return variants_status
    return is_in_forest


# %% [markdown]
# ### Is set target

# %%
def is_set_target_factory(all_variants):
    def is_set_target(env, scenario, state, agent_id, variants_status):
        for i, target in enumerate(targets):  # targets set in grammar 
            target_id = jnp.where(scenario.unit_team[agent_id]==1, 2*i, 2*i+1)  # even = allies targets | odd = enemies targets
            variants_status = variants_status.at[all_variants.index(f"is_set_target {target}")].set(jnp.where(jnp.all(scenario.targets_position[target_id]>=0), Status.SUCCESS, Status.FAILURE))
        return variants_status
    return is_set_target


# %% [markdown]
# ## Compute all variants

# %%
def compute_variants_factory(all_variants, n_agents):
    stand_eval = stand_factory(all_variants)
    move_eval = move_factory(all_variants)
    attack_eval = attack_factory(all_variants)
    follow_map_eval = follow_map_factory(all_variants)
    heal_eval = heal_factory(all_variants)
    go_to_eval = go_to_factory(all_variants)
    set_target_eval = set_target_factory(all_variants)
    debug_eval = debug_factory(all_variants)
    in_sight_eval = in_sight_factory(all_variants, n_agents)
    in_reach_eval = in_reach_factory(all_variants, n_agents)
    is_type_eval = is_type_factory(all_variants)
    is_dying_eval = is_dying_factory(all_variants)
    is_in_forest_eval = is_in_forest_factory(all_variants)
    is_set_target_eval = is_set_target_factory(all_variants)
    def compute_variants(env, scenario, state, rng, agent_id, variants_status, variants_action):
        move_rng, attack_rng, follow_map_rng, heal_rng, set_target_rng = random.split(rng, 5)
        variants_status, variants_action = stand_eval(env, scenario, state, agent_id, variants_status, variants_action)
        variants_status, variants_action = move_eval(env, scenario, state, move_rng, agent_id, variants_status, variants_action)
        variants_status, variants_action = attack_eval(env, scenario, state, attack_rng, agent_id, variants_status, variants_action)
        if "follow_map" in actions:
            variants_status, variants_action = follow_map_eval(env, scenario, state, follow_map_rng, agent_id, variants_status, variants_action)
        if "heal" in actions:
            variants_status, variants_action = heal_eval(env, scenario, state, heal_rng, agent_id, variants_status, variants_action)
        variants_status, variants_action = go_to_eval(env, scenario, state, agent_id, variants_status, variants_action)
        variants_status, variants_action = set_target_eval(env, scenario, state, set_target_rng, agent_id, variants_status, variants_action)
        if "debug" in actions:
            variants_status, variants_action = debug_eval(env, scenario, state, agent_id, variants_status, variants_action)
        variants_status = in_sight_eval(env, scenario, state, agent_id, variants_status)
        variants_status = in_reach_eval(env, scenario, state, agent_id, variants_status)
        variants_status = is_type_eval(env, scenario, state, agent_id, variants_status)
        variants_status = is_dying_eval(env, scenario, state, agent_id, variants_status)
        if "is_in_forest" in conditions:
            variants_status = is_in_forest_eval(env, scenario, state, agent_id, variants_status)
        variants_status = is_set_target_eval(env, scenario, state, agent_id, variants_status)
        return variants_status, variants_action
    return compute_variants


# %% [markdown]
# # get action

# %%
def eval_bt(predecessors, parents, passing_nodes, variant_ids, variants_status, variants_action):
    def eval_leaf(i, carry):
        s, a, a_id, passing = carry
        variant_id = variant_ids[i]
        has_not_found_action = jnp.logical_or(s != Status.SUCCESS, a.kind == NONE)
        is_valid_from_sequence = jnp.logical_and(predecessors[i] == Parent.SEQUENCE, s != Status.FAILURE)
        is_valid_from_fallback = jnp.logical_and(predecessors[i] == Parent.FALLBACK, s != Status.SUCCESS)
        is_valid_from_root = predecessors[i] == Parent.NONE
        
        is_valid = jnp.logical_or(jnp.logical_or(is_valid_from_sequence, is_valid_from_fallback), is_valid_from_root)
        is_valid = jnp.logical_and(is_valid, variant_id != -1)  # not an empty leaf (from fixed size)
        condition = jnp.logical_and(jnp.logical_and(has_not_found_action, is_valid), passing <= 0)

        passing_if_FAILURE_in_sequence = jnp.logical_and(parents[i] == Parent.SEQUENCE, variants_status[variant_id] == Status.FAILURE)
        passing_if_SUCCESS_in_failure = jnp.logical_and(parents[i] == Parent.FALLBACK, variants_status[variant_id] == Status.SUCCESS)

        if_passing = jnp.logical_and(condition, jnp.logical_or(passing_if_FAILURE_in_sequence, passing_if_SUCCESS_in_failure))
        passing = jnp.where(if_passing, passing_nodes[i], passing-1)

        s = jnp.where(condition, variants_status[variant_id], s)
        a = variants_action[variant_id].where(condition, a)
        a_id = jnp.where(condition, variant_id, a_id)
        return s, a, a_id, passing
    return eval_leaf


# %%
def get_action_factory(all_variants, n_agents, bt_max_size):
    n_variants = len(all_variants)
    compute_variants = compute_variants_factory(all_variants, n_agents)
    def get_action(env, scenario, state, rng, predecessors, parents, passing_nodes, variant_ids, agent_id):  # for one agent
        variants_status, variants_action = compute_variants(env, scenario, state, rng, agent_id, jnp.zeros(n_variants), Action.from_shape((n_variants,)))
        eval_leaf = eval_bt(predecessors, parents, passing_nodes, variant_ids, variants_status, variants_action)
        carry = Status.NONE, NONE_ACTION, -1, 0
        s, a, a_id, p = fori_loop(0, bt_max_size, eval_leaf, carry)
        return a.where(jnp.logical_and(s == Status.SUCCESS, a.kind != NONE), STAND_ACTION), jnp.where(jnp.logical_and(s == Status.SUCCESS, a.kind != NONE), a_id, all_variants.index("stand"))
    return get_action
