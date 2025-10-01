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
import numpy as np
from typing import Tuple, Optional

# %% [markdown]
# # Extracted from https://github.com/EvolutionGym/evogym to add seeding of the rng
#
# MIT License
#
# Copyright (c) 2022 jagdeepsb
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

# %%
VOXEL_TYPES = {
    'EMPTY': 0,
    'RIGID': 1,
    'SOFT': 2,
    'H_ACT': 3,
    'V_ACT': 4,
    'FIXED': 5,
}


def get_uniform(x: int) -> np.ndarray:
    """
    Return a uniform distribution of a given size.

    Args:
        x (int): size of distribution. Must be positive.
    
    Returns:
        np.ndarray: array representing the probability distribution.
    """
    assert x > 0, f"Invalid size {x} for uniform distribution. Must be positive."
    return np.ones((x)) / x

def draw(rng, pd: np.ndarray) -> int:
    """
    Sample from a probability distribution.

    Args:
        pd (np.ndarray): array representing the relative probability of sampling each element. Entries must be non-negative and sum to a non-zero value. Must contain at least one element.
    
    Returns:
        int: sampled index.
    """
    pd_copy = pd.copy()
    if not isinstance(pd_copy, np.ndarray):
        pd_copy = np.array(pd_copy)
        
    assert pd_copy.size > 0, f"Invalid size {pd_copy.size} for probability distribution. Must contain at least one element."
    assert np.all(pd_copy >= 0), f"Invalid probability distribution {pd_copy}. Entries must be non-negative."
    assert np.sum(pd_copy) > 0, f"Invalid probability distribution {pd_copy}. Entries must sum to a non-zero value."
    
    pd_copy = pd_copy / pd_copy.sum()

    rand = rng.uniform(0, 1)
    sum = 0
    for i in range(pd_copy.size):
        sum += pd_copy[i]
        if rand <= sum:
            return i

def sample_robot(
    rng,
    robot_shape: Tuple[int, int], 
    pd: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a randomly sampled robot of a particular size.

    Args:
        rng: numpy rng
        robot_shape (Tuple(int, int)): robot shape to sample `(h, w)`.
        pd (np.ndarray): `(5,)` array representing the relative probability of sampling each robot voxel (empty, rigid, soft, h_act, v_act). Defaults to a custom distribution. (default = None)
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: randomly sampled (valid) robot voxel array and its associated connections array.
        
    Throws:
        If it is not possible to sample a connected robot with at least one actuator.
    """
    
    h_act, v_act, empty = VOXEL_TYPES['H_ACT'], VOXEL_TYPES['V_ACT'], VOXEL_TYPES['EMPTY']
    
    if pd is not None:
        assert pd.shape == (5,), f"Invalid probability distribution {pd}. Must have shape (5,)."
        if pd[h_act] + pd[v_act] == 0:
            raise ValueError(f"Invalid probability distribution {pd}. Must have a non-zero probability of sampling an actuator.")
        if sum(pd) - pd[empty] == 0:
            raise ValueError(f"Invalid probability distribution {pd}. Must have a non-zero probability of sampling a non-empty voxel.")
    
    done = False

    while (not done):

        if pd is None:
            pd = get_uniform(5)
            pd[0] = 0.6

        robot = np.zeros(robot_shape)
        for i in range(robot.shape[0]):
            for j in range(robot.shape[1]):
                robot[i][j] = draw(rng, pd)

        if is_connected(robot) and has_actuator(robot):
            done = True

    return robot

def _is_in_bounds(x: int, y: int, width: int, height: int) -> bool:
    """
    Returns whether or not a certain index is within bounds.

    Args:
        x (int): x pos.
        y (int): y pos.
        width (int): max x.
        height (int): max y.
    """
    if x < 0:
        return False
    if y < 0:
        return False
    if x >= width:
        return False
    if y >= height:
        return False
    return True
    
def _recursive_search(x: int, y: int, connectivity: np.ndarray, robot: np.ndarray) -> None:
    """
    Performs a floodfill search.
    
    Args:
        x (int): x pos.
        y (int): y pos.
        connectivity (np.ndarray): array to be filled in during floodfill.
        robot (np.ndarray): array specifing the voxel structure of the robot.
    """
    if robot[x][y] == 0:
        return
    if connectivity[x][y] != 0:
        return

    connectivity[x][y] = 1

    for x_offset in [-1, 1]:
        if _is_in_bounds(x + x_offset, y, robot.shape[0], robot.shape[1]):
            _recursive_search(x + x_offset, y, connectivity, robot)

    for y_offset in [-1, 1]:
        if _is_in_bounds(x, y + y_offset, robot.shape[0], robot.shape[1]):
            _recursive_search(x, y + y_offset, connectivity, robot)

def is_connected(robot: np.ndarray) -> bool:
    """
    Returns whether or not a certain robot is connected by running floodfill.

    Args:
        robot (np.ndarray): array specifing the voxel structure of the robot.
    
    Returns:
        bool: whether or not the robot is connected.
    """
    is_found = np.zeros(robot.shape)

    start = None
    for i in range(robot.shape[0]):
        if start:
            break
        for j in range(robot.shape[1]):
            if robot[i][j] != 0:
                start = (i, j)
                break

    if start == None:
        return False

    connectivity = np.zeros(robot.shape)
    _recursive_search(start[0], start[1], connectivity, robot)

    for i in range(robot.shape[0]):
        for j in range(robot.shape[1]):
            if robot[i][j] != 0 and connectivity[i][j] != 1:
                return False

    return True

def has_actuator(robot: np.ndarray) -> bool:
    """
    Returns whether or not a certain robot has an actuator.
    Args:
        robot (np.ndarray): array specifing the voxel structure of the robot.
    
    Returns:
        bool: whether or not the robot has an actuator.
    """
    for i in range(robot.shape[0]):
        for j in range(robot.shape[1]):
            if robot[i][j] == 3 or robot[i][j] == 4:
                return True

    return False


# %%
