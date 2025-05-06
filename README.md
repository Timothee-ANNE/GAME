# README

Repository containing the code used to generate the results of "Adversarial Coevolutionary Illumination with Generational Adversarial MAP-Elites" using the proposed Generational Adversarial MAP-Elites (GAME) algorithm.
All Python scripts are formatted to be opened in a JupyterLab notebook using the Jupytext extension.

The experiments were run on an NVIDIA GeForce RTX 4090 or NVIDIA RTX 6000 Ada Generation for the Parabellum evaluations (including CLIP), but the main algorithm GAME runs on CPU with no acceleration.
Expected time for the paper parameters (20 generations, 100k evaluations/generation): 72h-120h + 12h-48h for the intergenerational tournament.

## Videos
* figure_1.mp4: We provide an enriched video for Figure 1, with additional behaviors.
* a.txt to o.txt: Behavior trees used in the different duels and the list of atomics used.

## Main
* GAME.py: Main file containing the GAME algorithm and the interface to the Parabellum game. To avoid adding discrepancies, the code contains unused parameters, i.e., the current implementation of Parabellum allows the parametrization of the units' positions and types, but the current setting ignores this feature and fixes the initial position and unit types. The current settings are those used to obtain the paper's results.
* plots.py: File containing the functions to plot the different figures.
* requirements.txt: Pip freeze requirements. Python=3.10.13. Requires a CUDA compatible environment. Not everything is needed, but everything you need is included.

## Misc 
* grid_plot.py: Script to create an HTML page to visualize videos with extra information.
* plot.py: Script to plot data with statistical tests.
* utils.py: Load/save functions.
  
## Parabellum
* atomics.py: Actions/conditions atomic functions in JAX.
* BT_array.py: Translates a BT expression into a BT array that is JAX/Jit compatible.
* BT_tree.py: Translates a BT expression into a BT tree (anytree).
* clip.py: CLIP VLM implementation.
* env.py: Main Parabellum script (step function).
* eval_bit_bts.py: Main script to take two BTs, run them in Parabellum, and return collected data.
* grammar.py: Lark grammar to define the BT.
* tps.py: Defines the terrain types (not heavily used in this implementation).
