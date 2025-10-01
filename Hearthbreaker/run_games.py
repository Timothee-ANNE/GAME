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

# %%
import sys
sys.path.append("hearthbreaker")  
from hearthbreaker.agents.basic_agents import RandomAgent
from hearthbreaker.agents.trade_agent import TradeAgent
from hearthbreaker.cards.heroes import hero_for_class
from hearthbreaker.constants import CHARACTER_CLASS
from hearthbreaker.engine import Game, Deck, card_lookup, card_table
from hearthbreaker.cards import *
import pickle
import numpy as np
import argparse


# %%
def load_deck(filename, character_class):
    cards = []

    with open(filename, "r") as deck_file:
        contents = deck_file.read()
        items = contents.splitlines()
        for line in items[0:]:
            parts = line.split(" ", 1)
            count = int(parts[0])
            for i in range(0, count):
                card = card_lookup(parts[1])
                if card.character_class != CHARACTER_CLASS.ALL and card.character_class != character_class:
                    print("Class error!", card.character_class, character_class)
                    return None 
                cards.append(card)

    if len(cards) != 30:
        print("Deck must have 30 cards!")
        return None

    return Deck(cards, hero_for_class(character_class))


# %%
def get_behavior(deck):
    mana = [c.mana for c in deck.cards]
    return np.array([np.mean(mana), np.std(mana)])


# %%
def write_error(path, error, name="error"):
    with open(path + name + ".txt", "w") as f:
        f.write(str(error))

def eval_decks(path, n_replications, seed, task_class, candidate_class, generation, agent_name):
    assert agent_name in ["Random", "Trader"]    
    n_replications = int(n_replications)
    task_class = int(task_class)
    candidate_class = int(candidate_class)
    generation = generation

    # need copies of the deck for the two starting games
    task_deck1 = load_deck(path+"task_deck.txt", task_class)
    candidate_deck1 = load_deck(path+"candidate_deck.txt", candidate_class)
    task_deck2 = load_deck(path+"task_deck.txt", task_class)
    candidate_deck2 = load_deck(path+"candidate_deck.txt", candidate_class)
    
    if task_deck1 is None or candidate_deck1 is None or candidate_deck2 is None or task_deck2 is None:
        pass
            
    if generation == "red":
        blue_deck1, red_deck1 = task_deck1, candidate_deck1
        blue_deck2, red_deck2 = task_deck2, candidate_deck2
    else:
        blue_deck1, red_deck1 = candidate_deck1, task_deck1
        blue_deck2, red_deck2 = candidate_deck2, task_deck2

    # run 
    H_diff = np.zeros(n_replications)
    agent = RandomAgent() if agent_name == "Random" else TradeAgent()

    try:
        red_starting_game = Game([red_deck1, blue_deck1], [agent, agent], seed, 0)
        blue_starting_game = Game([red_deck2, blue_deck2], [agent, agent], seed, 1)

        for rep in range(n_replications):
            new_game = red_starting_game.copy() if rep % 2 == 0 else blue_starting_game.copy()
            new_game.start()
            h_red, h_blue = new_game.players[new_game.first_player].hero.health, new_game.players[1-new_game.first_player].hero.health
            del new_game
            H_diff[rep] = h_red - h_blue  # like Fontaine 2019, fitness = health one - health of the other 
    
        del red_starting_game
        del blue_starting_game
    except:
        pass
        
    data = {"f_r": np.mean(H_diff), "blue_behavior": get_behavior(blue_deck1), "red_behavior": get_behavior(red_deck1)}
    with open(path+"evaluation.pk", "wb") as f:
        pickle.dump(data, f)


# %%
def eval_decks_local(path, n_replications, seed, task_class, candidate_class, generation, agent_name):
    assert agent_name in ["Random", "Trader"]    
    n_replications = int(n_replications)
    task_class = int(task_class)
    candidate_class = int(candidate_class)
    generation = generation

    # need copies of the deck for the two starting games
    task_deck1 = load_deck(path+"task_deck.txt", task_class)
    candidate_deck1 = load_deck(path+"candidate_deck.txt", candidate_class)
    task_deck2 = load_deck(path+"task_deck.txt", task_class)
    candidate_deck2 = load_deck(path+"candidate_deck.txt", candidate_class)
    
    if task_deck1 is None or candidate_deck1 is None or candidate_deck2 is None or task_deck2 is None:
        pass
            
    if generation == "red":
        blue_deck1, red_deck1 = task_deck1, candidate_deck1
        blue_deck2, red_deck2 = task_deck2, candidate_deck2
    else:
        blue_deck1, red_deck1 = candidate_deck1, task_deck1
        blue_deck2, red_deck2 = candidate_deck2, task_deck2

    # run 
    H_diff = np.zeros(n_replications)
    agent = RandomAgent if agent_name == "Random" else TradeAgent


    red_starting_game = Game([red_deck1, blue_deck1], [agent(), agent()], seed, 0)
    blue_starting_game = Game([red_deck2, blue_deck2], [agent(), agent()], seed, 1)

    for rep in range(n_replications):
        new_game = red_starting_game.copy() if rep % 2 == 0 else blue_starting_game.copy()
        new_game.start()
        h_red, h_blue = new_game.players[new_game.first_player].hero.health, new_game.players[1-new_game.first_player].hero.health
        del new_game
        H_diff[rep] = h_red - h_blue  # like Fontaine 2019, fitness = health one - health of the other 

    del red_starting_game
    del blue_starting_game

        
    return {"f_r": np.mean(H_diff), "blue_behavior": get_behavior(blue_deck1), "red_behavior": get_behavior(red_deck1)}


# %%
def main():
    parser = argparse.ArgumentParser(description='Execute eval_decks function')
    
    parser.add_argument('path', type=str, help='Path parameter')
    parser.add_argument('n_replications', type=str, help='Number of replications')
    parser.add_argument('seed', type=str, help='Random seed')
    parser.add_argument('task_class', type=str, help='Task class')
    parser.add_argument('candidate_class', type=str, help='Candidate class')
    parser.add_argument('generation', type=str, help='Generation parameter')
    parser.add_argument('agent_name', type=str, help='Agent name')
    
    args = parser.parse_args()
    
    # Execute the function with the provided arguments
    eval_decks(
        path=args.path,
        n_replications=args.n_replications,
        seed=args.seed,
        task_class=args.task_class,
        candidate_class=args.candidate_class,
        generation=args.generation,
        agent_name=args.agent_name
    )

if __name__ == "__main__":
    main()
