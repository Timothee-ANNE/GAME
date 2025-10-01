# README

Repository containing the code used to generate the results of the conference paper "Generational Adversarial MAP-Elites for Multi-Agent Game Illumination" presented at ALIFE '25, Kyoto, Japan, October 06–10, and its extended version "Adversarial Coevolutionary Illumination with Generational Adversarial MAP-Elites" using the proposed Generational Adversarial MAP-Elites (GAME) algorithm.  
All Python scripts are formatted to be opened in a JupyterLab notebook using the Jupytext extension. All GAME runs and ablations/baselines can be executed entirely. The `plot_for_paper` scripts require datasets and serve more as an archived version.

The experiments were run on an NVIDIA GeForce RTX 4090 or NVIDIA RTX 6000 Ada Generation for the Parabellum evaluations (including CLIP). The main algorithm GAME runs on CPU with no acceleration.  
Expected runtime for the Parabellum experiments with paper parameters (20 generations, 100k evaluations per generation): 72h–120h plus 12h–48h for the intergenerational tournament.

## Videos
### Case study 1: Parabellum
- `figure_1.mp4`: Enriched video for Figure 1, with additional behaviors.  
- `a.txt` to `o.txt`: Behavior trees used in the different duels and the list of atomics.  

### Case study 2: Wrestling
- `figure_8.mp4`: Video for Figure 8.

## Requirements
- `requirements.txt`: Pip freeze requirements. Python=3.10.13. Requires a CUDA-compatible environment. Not everything is required, but everything you need is included.  

## Parabellum
### Main
- `GAME.py`: Main file containing the GAME algorithm and the interface to the Parabellum game. To avoid discrepancies, the code contains unused parameters. For example, Parabellum allows parametrization of the units' positions and types, but the current setting ignores this feature and fixes the initial positions and unit types. These are the settings used to obtain the paper's results.  
- `plots.py`: Functions to plot the figures.  

### Misc
- `grid_plot.py`: Creates an HTML page to visualize videos with extra information.  
- `plot.py`: Plots data with statistical tests.  
- `utils.py`: Load/save functions.  

### Parabellum
- `atomics.py`: Action/condition atomic functions in JAX.  
- `BT_array.py`: Translates a BT expression into a BT array that is JAX/JIT-compatible.  
- `BT_tree.py`: Translates a BT expression into a BT tree (anytree).  
- `clip.py`: CLIP VLM implementation.  
- `env.py`: Main Parabellum script (step function).  
- `eval_bit_bts.py`: Takes two BTs, runs them in Parabellum, and returns collected data.  
- `grammar.py`: Lark grammar to define the BT.  
- `tps.py`: Defines terrain types (not heavily used in this implementation).  

## Wrestling
### Main
- `GAME.py`: GAME implementation.  
- `misc_plot.py`: Plot functions.  
- `plot_for_paper.py`: Creates the paper's figures. Requires running `GAME.py`.  
- `Random.py`: Random ablation for the main comparison.  
- `utils.py`: Load/save functions.  

### wrestlers
Custom environment from EvoGym (`envs/wrestling_env.py`) (https://github.com/EvolutionGym/evogym) and CLIP from ASAL (https://github.com/SakanaAI/asal).  

## Hearthbreaker
### Main
- `GAME.py`: GAME for the main comparison, similar to the other GAME implementation but without the VEM (you need to provide the absolute path to `GAME/Hearthbreaker/run_games.py` in the config of `exec_path`).  
- `ME.py`: ME ablation for the main comparison (you need to provide the absolute path to `GAME/Hearthbreaker/run_games.py` in the config of `exec_path`).  
- `misc_plot.py`: Plot functions.  
- `plot_for_paper.py`: Creates the paper's figures. Requires running `ME.py` and `GAME.py`.  
- `run_games.py`: Uses Hearthbreaker to evaluate two decks.  
- `utils.py`: Load/save functions.  

### heartbreaker
Copy from Hearthbreaker (https://github.com/danielyule/hearthbreaker) with some modifications:  
- `engine.py`: Fixes the seeds and the player order.  
- `agents/trade/trade.py`: Avoids infinite loops in the Trade Agent heuristic.  

```python

```
