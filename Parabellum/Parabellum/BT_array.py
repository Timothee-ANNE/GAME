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
import jax.numpy as jnp
from lark import Lark, Transformer
from flax.struct import dataclass

# %%
from grammar import * 


# %% [markdown]
# # The Array

# %%
@dataclass
class Parent:  # for behavior tree
    SEQUENCE: int = 1
    NONE: int = 0
    FALLBACK: int = -1


# %% [markdown]
# ## expr 2 array

# %%
def compute_right_siblings(n_nodes):
    a = np.array(n_nodes)
    a = -a
    a[0] = np.sum(-a)+a[0]
    return np.cumsum(a)


# %%
class Expr2array(Transformer):
    def __init__(self, all_variants):
        self.all_variants = all_variants
        
    def node(self, args):
        return args[0]

    def nodes(self, args):
        return args

    def condition(self, args):
        return [(Parent.NONE, Parent.NONE, None, self.all_variants.index(" ".join(args[0])))]

    def action(self, args):
        return [(Parent.NONE, Parent.NONE, None, self.all_variants.index(" ".join(args[0])))]

    def sequence(self, args):
        array = []
        n_nodes = [len(child_array) for child_array in args[0]]
        n_right_siblings = compute_right_siblings(n_nodes)
        for child_id, child_array in enumerate(args[0]):
            array += [(Parent.SEQUENCE if i == 0 else pred, Parent.SEQUENCE if parent == Parent.NONE else parent, n_right_siblings[child_id] if passing is None else passing, a) for i, (pred, parent, passing, a) in enumerate(child_array)]
        return array

    def fallback(self, args):
        array = []
        n_nodes = [len(child_array) for child_array in args[0]]
        n_right_siblings = compute_right_siblings(n_nodes)
        for child_id, child_array in enumerate(args[0]):
            array += [(Parent.FALLBACK if i == 0 else pred, Parent.FALLBACK if parent == Parent.NONE else parent, n_right_siblings[child_id] if passing is None else passing, a) for i, (pred, parent, passing, a) in enumerate(child_array)]
        return array

    def atomic(self, args):
        return args[0]

    def move(self, args):
        return ["move"] + args
        
    def attack(self, args):
        return ["attack"] + args
        
    def stand(self, args):
        return ["stand"] 
    
    def follow_map(self, args):
        return ["follow_map"] + args

    def heal(self, args):
        return ["heal"] + args

    def go_to(self, args):
        return ["go_to"] + args

    def set_target(self, args):
        return ["set_target"] + args
        
    def debug(self, args):
        return ["debug"] + args
        
    def in_sight(self, args):
        return ["in_sight"] + args
    
    def in_reach(self, args):
        return ["in_reach"] + args
        
    def is_type(self, args):
        return ["is_type"] + args
    
    def is_dying(self, args):
        return ["is_dying"] + args

    def is_in_forest(self, args):
        return ["is_in_forest"]
        
    def is_set_target(self, args):
        return ["is_set_target"] + args
        
    def qualifier(self, args):
        return str(args[0])

    def sense(self, args):
        return str(args[0])
        
    def direction(self, args):
        return str(args[0])
    
    def foe(self, args):
        return str(args[0])
        
    def friend(self, args):
        return str(args[0])

    def qualifier(self, args):
        return str(args[0])

    def unit(self, args):
        return str(args[0])  

    def any(self, args):
        return str(args[0])
        
    def margin(self, args):
        return str(args[0])

    def source(self, args):
        return str(args[0])
    
    def steps(self, args):
        return str(args[0])

    def self(self, args):
        return str(args[0])

    def threshold(self, args):
        return str(args[0])
    
    def target(self, args):
        return str(args[0])
        
expr2array_transformer = Expr2array(all_variants)

def expr2array(expr, size):
    A = expr2array_transformer.transform(expr)  # [(parent, atomic id)]
    
    parents = jnp.ones(size, dtype=jnp.int32) * Parent.NONE
    predecessors = jnp.ones(size, dtype=jnp.int32) * Parent.NONE
    atomics_id = jnp.ones(size, dtype=jnp.int32) * -1
    passings = jnp.zeros(size, dtype=jnp.int32)
    assert len(A) <= size, f"The expr has {len(A)} leaves which is larger than the defined size {size}."
    for i, (predecessor, parent, passing, atomic_id) in enumerate(A):
        predecessors = predecessors.at[i].set(predecessor)
        parents = parents.at[i].set(parent)
        passings = passings.at[i].set(0 if passing is None else passing) 
        atomics_id = atomics_id.at[i].set(atomic_id)
    return predecessors, parents, passings, atomics_id

def txt2array(txt, size):
    return expr2array(txt2expr(txt), size)

# %% [raw]
# txt = "S(S(C (in_sight friend any) :: C (in_sight foe any))::F(C (in_sight friend archer) :: C (in_sight foe archer)))"
# txt = "F(A (attack closest any) :: S (C (in_sight foe any) :: A (go_to A 0%)) :: A (stand))"
# #txt = "F(A (stand) :: A (move toward closest foe any))"
# #txt = "S(A (stand) :: A (stand))"
# #txt = "S(A(stand))"
# #txt = "A (go_to A 100%)"
# #txt = "A (stand)"
# txt2array(txt, 4)

# %%

# %%
