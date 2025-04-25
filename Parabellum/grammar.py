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
from lark import Lark, Transformer
from itertools import product, chain, combinations


# %% [markdown]
# # The Grammar

# %%
def set_product(sets):
    return [" ".join(a) for a in product( *tuple(vals for vals in sets))]


# %%
qualifiers = ["closest", "farthest", "weakest", "strongest", "random"]
senses = ["toward", "away_from"]
directions = ["north", "south", "east", "west"]
unit_types = ["spearmen", "archer", "cavalry", "healer", "grenadier"]
source = ["me_from_them", "them_from_me"]
steps = ["0", "1", "2", "3"]
any_ = ["any"]
thresholds = ["25%", "50%", "75%"]
margins = ["0%", "25%", "50%", "100%"]
unit_any = unit_types + any_ 
targets = ["A", "B", "C", "D", "E"] #, "C", "D", "E"]

actions = {
    "attack": set_product([["attack"], qualifiers, unit_any]),
    "move":  set_product([["move"], senses, qualifiers, ["foe", "friend"], unit_any]),
    "stand": ["stand"],
    #"follow_map": set_product([["follow_map"], senses, margins]),
    "heal": set_product([["heal"], qualifiers, unit_any]),
    "go_to": set_product([["go_to"], targets, margins]),
    "set_target": set_product([["set_target"], targets, qualifiers, ["foe", "friend"], unit_any]),
    #"debug": set_product([["debug"], directions]),
}

conditions = {
    "in_sight": set_product([["in_sight"], ["foe", "friend"], unit_any]),
    "in_reach": set_product([["in_reach"], ["foe", "friend"], source, steps, unit_any]),
    "is_type": set_product([["is_type"], unit_types]),
    "is_dying": set_product([["is_dying"], ["self", "foe", "friend"], thresholds]),
    #"is_in_forest": ["is_in_forest"],
    "is_set_target": set_product([["is_set_target"], targets])
}

atomics = {"A": actions, "C": conditions}

# %%
all_variants = []
for atomic in atomics.values():
    for variants in atomic.values():
        all_variants += variants

# %%
bt_grammar = Lark(f'''?start: node

%import common.WS
%ignore WS

nodes : node ("::" node | "|>" node)*
node  :
    | sequence
    | fallback
    | action
    | condition

sequence  : "S" "(" nodes ")"
fallback  : "F" "(" nodes ")"
action    : "A" "(" atomic ")"
condition : "C" "(" atomic ")"

atomic :
    | move
    | attack
    | stand
    | follow_map
    | go_to
    | set_target 
    | heal
    | debug
    | in_sight
    | in_reach
    | is_type
    | is_dying 
    | is_in_forest 
    | is_set_target
    
    
move       : "move" sense qualifier (foe | friend) (unit | any)
attack     : "attack" qualifier (unit |any)
stand      : "stand"
follow_map : "follow_map" sense margin
heal       : "heal" qualifier (unit |any)
go_to      : "go_to" target margin
set_target : "set_target" target qualifier (foe | friend) (unit | any)
debug      : "debug" direction
in_sight  : "in_sight" (foe | friend) (unit | any) 
in_reach  : "in_reach" (foe | friend) source steps (unit | any) 
is_type   : "is_type" unit 
is_dying  : "is_dying" (self | foe | friend) threshold
is_in_forest : "is_in_forest"
is_set_target  : "is_set_target" target
qualifier : /{"|".join(qualifiers)}/
margin  : /{"|".join(margins)}/
unit      : /{"|".join(unit_types)}/
sense     : /{"|".join(senses)}/
foe       : /foe/
friend    : /friend/
self      : /self/
any       : /any/
direction : /{"|".join(directions)}/
source    : /{"|".join(source)}/
steps     : /{"|".join(steps)}/
threshold : /{"|".join(thresholds)}/
target    : /{"|".join(targets)}/
''')


# %%
def txt2expr(txt):
    return bt_grammar.parse(txt)


# %% [markdown]
# # compute variants from unit_types 

# %%
def choose_targets(bt_txt, targets):
    targets_txt = "any" if len(targets) == 0 else " or ".join([t for t in targets])
    return bt_txt.replace("any", targets_txt)

def compute_all_variants(bt_txt, unit_types):
    if "any" in bt_txt:
        subsets = [{}] + [set([unit_type]) for unit_type in unit_types]
        variants = [choose_targets(bt_txt, subset) for subset in subsets]
        return subsets, variants
    else:
        return [None], [bt_txt]

def find_bt_variant(x, subsets, variants):
    assert(x in subsets)
    return variants[subsets.index(x)]       

def set_default(bt_txt, default):
    return f"F ( S( C (in_sight foe any) :: {bt_txt}) :: {default})"
