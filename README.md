# README


Repository countaining the code used to generate the result of "Adversarial Co-evolution Quality Diversity" using the proposed  Generational Adversarial MAP-Elites (GAME) algorithm.
All python scripts are formatted to be open in a JupyterLab notebook using the Jupytext extension. 

The experiments were run on a NVIDIA GeForce RTX 4090 or NVIDIA RTX 6000 Ada Generation for the Parabellum evaluations (including CLIP) but the main algorithm GAME runs on CPU with no acceleration. 
Expected time for the paper paramaters (20 generations, 100k evaluations/genration) 72h-120h + 12h-48h for the intergenerational-tournament.

## Videos
* figure_1.mp4: We propose an enriched video for the Figure 1, with additional behaviors.
* a.txt to o.txt: behavior trees used in the different duels and the list of atomics used. 

## Main
* GAME.py: main file containing the GAME algorithm and the interface to Parabellum game.  To avoid adding discrepancy, the code contains unused parameters, i.e., the current implementation of Parabellum allows the parametrization of the units positions and types, but the current setting ignores this feature and fixes the initial position and unit types. The current settings are those used to obtain the paper's results.
* plot_figures.py: file containing the functions to plot the different figures
* requirements.txt: Pip freeze requirements. Require a CUDA compatible environment. Not everything is needed but everything you need is included.

## Misc 
* grid_plot.py: scipt to create html page to look at videos with extra information
* plot.py: script to plot data with statistical tests
* utils.py: load/save functions
  
## Parabellum
* atomics.py: actions/conditions atomics function in JAX
* BT_array: translate a BT expression into a BT array that is JAX/Jit compatible
* BT_tree: translate a BT expression into a BT tree (anytree)
* clip.py: CLIP VLM
* env.py: main Parabellum script (step function)
* eval_bit_bts: main script to take two BTs, run them in Parabellum, and return collected data
* grammar.py: Lark grammar to define the BT
* tps.py: defines the Terrain types (not really used here)
