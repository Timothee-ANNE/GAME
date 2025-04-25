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
from anytree import NodeMixin, RenderTree, Node
from copy import deepcopy

# %%
from grammar import txt2expr, atomics


# %% [markdown]
# # The BT

# %%
class Found(Exception): 
    '''
    Exception for BT.visit_tree(node_id)
    '''
    def __init__(self, x): 
        super().__init__("")
        self.value = x 


class BT(Node):
    leaves = ["A", "C"]
    flow_nodes = ["F", "S"]
    atomics = atomics

    def random_atomic(rng, leaf_type=None, atomic_type=None):
        leaf_type = rng.choice(BT.leaves) if leaf_type is None else leaf_type
        atomic_type = rng.choice(list(BT.atomics[leaf_type].keys())) if atomic_type is None else atomic_type
        return rng.choice(BT.atomics[leaf_type][atomic_type])
    
    def random(rng, max_depth, max_children, leaf_type=None):
        max_depth = rng.integers(0, max_depth+1)
        if max_depth == 0 or max_children < 2:  # leaf
            leaf_type = rng.choice(BT.leaves) if leaf_type is None else leaf_type
            leaf_atomic = BT.random_atomic(rng, leaf_type)
            return BT(leaf_type, atomic=leaf_atomic).clean()
        else:
            flow_node_type = rng.choice(BT.flow_nodes)
            n_children = rng.integers(2, max_children+1)
            children = [ BT.random(rng, max_depth-1, max_children) for _ in range(n_children)] 
            return BT(flow_node_type, children=children).clean()
    
    def __init__(self, name, atomic=None, parent=None, children=None):
        super(BT, self).__init__(name)
        assert name in ["S", "F", "D", "A", "C"]
        if self.name in ["A", "C"]:
            assert atomic is not None
        self.atomic = atomic
        self.parent = parent
        if children:
            self.children = children

    def n_leaves(self):
        if self.name in ["A", "C"]:
            return 1
        else:
            return np.sum([child.n_leaves() for child in self.children])
    
    def is_valid(self):
        if self.name in ["A", "C"]:
            return True
        else:
            if len(self.children) == 0:
                return False 
            for child in self.children:
                if not child.is_valid():
                    return False
            return True 

    def compute_nodes_id(self):
        def aux(bt, counter):
            bt.id = counter
            c = counter + 1
            for child in bt.children:
                c = aux(child, c)
            return c
        aux(self, 0)
        
    def pretty_print(self, intensity=None, colormap="cividis", show_node_id=False):
        for i, (pre, fill, node) in enumerate(RenderTree(self)):
            treestr = u"%s%s" % (pre, node.name)
            txt = treestr.ljust(8) + f" {node.atomic if node.atomic else ''}{' (' + str(i) + ')' if show_node_id else ''}"
            if intensity is not None:
                fore_fromrgba(txt, int_to_color(intensity[i], vmax=1, vmin=0, cmap=colormap))
            else:
                print(txt)

    def to_pretty_txt(self, show_node_id=False):
        txt = "" 
        for i, (pre, fill, node) in enumerate(RenderTree(self)):
            treestr = u"%s%s" % (pre, node.name)
            txt += treestr.ljust(8) + f" {node.atomic if node.atomic else ''}{' (' + str(i) + ')' if show_node_id else ''}\n"
        return txt
    
    def to_txt(self, root=True):
        if self.name == "A":
            return f"A ({self.atomic})"
        elif self.name == "C":
            return f"C ({self.atomic})"
        children_expr = ":: ".join([child.to_txt(False) for child in self.children])
        if self.name == "D":
            return f"D ({self.atomic}, {children_expr})"
        elif self.name == "F":
            return  f"F ({children_expr})"
        elif self.name == "S":
            return f"S ({children_expr})"

    def to_expr(self):
        return txt2expr(self.to_txt())
    
    def visit_tree(self, node_id):
        """ 
        returns the node with id == node_id in the depth visit pattern
        """
        def aux(bt, counter):
            if counter == 0:
                raise Found(bt)
            c = counter - 1
            for child in bt.children:
                c = aux(child, c)
            return c
        try:
            aux(self, node_id)
        except Found as e:
            return e.value 
        return None
    
    def random_subtree(self, rng, exclude_root=True):
        '''
        return an error if exclude_root is True and the tree is of size 1
        '''
        picked_id = rng.integers(exclude_root, self.size)
        subtree = self.visit_tree(picked_id)
        assert subtree is not None
        return subtree

    def mutation(self, rng, mutation_probas):
        '''
        returns a new BT tree
        node mutation: change the node to a random one of the same type (i.e., flow/leaf). Does not change the size of the tree
        node deletion: delete a subtree while keeping the tree valid (none empty children)
        node addtion: add a leaf 
        '''
        r = rng.random()
        new_bt = deepcopy(self)
        if (r < mutation_probas['deletion']) and new_bt.size > 1:
            selected_node = new_bt.random_subtree(rng, exclude_root=True)
            selected_node.delete()
            mutation_type = "deletion"
        elif r < mutation_probas['deletion'] + mutation_probas['weak mutation']:  # mutation
            selected_node = new_bt.random_subtree(rng, exclude_root=False)
            selected_node.mutate_node(rng, False)
            mutation_type = "weak mutation"
        elif r < mutation_probas['deletion'] + mutation_probas['weak mutation'] + mutation_probas['strong mutation']:  # mutation
            selected_node = new_bt.random_subtree(rng, exclude_root=False)
            selected_node.mutate_node(rng, True)
            mutation_type = "strong mutation"
        else:  # addition
            selected_node = new_bt.random_subtree(rng, exclude_root=False)
            selected_node.add_sibling(rng)
            mutation_type = "addition"
        return new_bt.clean(), mutation_type
    
    def mutate_node(self, rng, strong):
        '''
        change the node to a random one of the same type (flow/leaf)
        does not change the size of the tree
        '''
        if self.name in self.leaves:
            if strong:
                self.name = rng.choice(self.leaves)
                self.atomic = BT.random_atomic(rng, self.name)
            else:
                tmp = self.atomic
                self.atomic = BT.random_atomic(rng, self.name, self.atomic.split(" ")[0])
        elif self.name == "F":
            self.name = "S"
        elif self.name == "S":
            self.name == "F"

    def delete(self):
        node2del = self
        while node2del.parent is not None and node2del.parent.name == "D":  # if we supress the child of a decorator, we also del the decorator
            node2del = node2del.parent 
        node2del.parent = None
            
    def add_sibling(self, rng):
        new_sibling = BT.random(rng, 1, 2)
        new_sibling.parent = self.parent

    def crossover(self, rng, bt2):
        '''
        returns a new BT as a copy of self with a subtree switched by a subtree of bt2
        '''
        child = deepcopy(self)
        sub_bt1 = child.random_subtree(rng, False)
        sub_bt2 = deepcopy(bt2.random_subtree(rng, False))
        sub_bt2.parent = sub_bt1.parent
        sub_bt1.parent = None
        return child.clean()
    
    def clean(self):
        '''
        returns a canonique BT with the same behavior 
        '''
        return self.remove_doubles_action_siblings()

    def remove_doubles_action_siblings(self):
        # Do not change atomics
        if self.name in ["A", "C"]:
            return self
        # apply recursively on children
        children = [child.remove_doubles_action_siblings() for child in self.children]
        cleaned_children = []
        children_ids = set()
        for child in children:
            # remove double child
            child_id = child.to_txt()
            if child_id not in children_ids:
                children_ids.add(child_id)
                cleaned_children.append(child)
            else:
                child.delete()
        self.children = cleaned_children
        if self.name == "D":
            return self
        # a flow node with one child is equivalent to the child
        elif len(cleaned_children) == 1:
            self.delete
            return cleaned_children[0]
        return self


# %% [markdown]
# ## expr to tree

# %%
class Expr2tree(Transformer):
    
    def node(self, args):
        return args[0]

    def nodes(self, args):
        return args

    def condition(self, args):
        return BT('C', atomic=" ".join(args[0]))

    def action(self, args):
        return BT('A', atomic=" ".join(args[0]))

    def sequence(self, args):
        return BT('S', children=args[0])

    def fallback(self, args):
        return BT('F', children=args[0])

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
        
    def margin(self, args):
        return str(args[0])
        
    def unit(self, args):
        return str(args[0])  

    def any(self, args):
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
        
expr2tree_transformer = Expr2tree()

def expr2tree(expr):
    return expr2tree_transformer.transform(expr).clean()

def txt2tree(txt):
    return expr2tree(txt2expr(txt))

# %% [raw]
# bt1_txt = """F(
#         S( C (is_set_target A) :: F(A (attack random any) :: A (go_to A 25%)))::
#         S(C (in_sight foe any) :: A (set_target A closest foe any))::
#         A (move away_from closest friend any)
#     )
#     """
# bt1_expr = txt2expr(bt1_txt)
# bt = expr2tree(bt1_expr)
# bt.pretty_print()

# %%
