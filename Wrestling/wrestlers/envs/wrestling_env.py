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
from gymnasium import spaces
from evogym import EvoWorld, EvoSim, EvoViewer
from evogym.envs import EvoGymBase

import numpy as np
import os
import matplotlib.pyplot as plt
import io
from PIL import Image

# %%
from matplotlib.path import Path

# %%
wrestler_materials = {
    0: "empty", # inisible
    1: "rigid",  # black 
    2: "soft",  # gray
    3: "in-phase horizontal",  # orange
    4: "in-phase vertical",  # blue
    5: "antiphase horizontal",  # orange
    6: "antiphase vertical",  # blue
}

colors = {
    1.: "black",
    2.: "lightgray",
    3.: "orange",
    4.: "royalblue",
    5.: "gold",
    6.: "skyblue",
}

rgb_colors = {
    1.: np.array([1, 1, 1]),
    2.: np.array([211, 211, 211]),
    3.: np.array([255, 165, 0]),
    4.: np.array([65, 105, 225]),
    5.: np.array([255, 215, 0]),
    6.: np.array([135, 206, 235]),
}


# %%
def remove_phase(wrestler):
    body = np.copy(wrestler)
    body[body>=5] -= 2
    return body

def get_voxels(wrestler):
    Corners = {}
    voxels = []
    for i in range(wrestler.shape[0]):
        for j in range(wrestler.shape[1]):
            if wrestler[i,j]:
                y, x = 0.5-i, 0.5+j
                voxel = []
                a, b = 0, 0
                add_top_left_twin = (i > 0 and not wrestler[i-1,j]) and (j > 0 and not wrestler[i,j-1])
                add_top_right_twin = (i > 0 and not wrestler[i-1,j]) and ( j < wrestler.shape[1]-1 and not wrestler[i,j+1])
                for dx, dy in [(-0.5, 0.5), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]:
                    key = (x+dx, y+dy, 0)  
                    if key not in Corners:
                        Corners[key] = len(Corners)
                    else:  # key was already present but it's in diagonal
                        if (dx, dy) == (-0.5, 0.5) and add_top_left_twin:
                            key = (x+dx, y+dy, 1)
                            Corners[key] = len(Corners)
                            a = 1
                        elif (dx, dy) == (0.5, 0.5) and add_top_right_twin:
                            key = (x+dx, y+dy, 1)
                            Corners[key] = len(Corners)
                            b = 1
                for dx, dy, twin in [(-0.5, 0.5, a), (0.5, 0.5, b), (0.5, -0.5, 0), (-0.5, -0.5, 0)]:
                    key = (x+dx, y+dy, twin)
                    voxel.append(Corners[key])
                voxels.append((i, j, voxel))
    return voxels
    
class WrestlingEnvClass(EvoGymBase):
    def __init__(self, blue_wrestler, red_wrestler, T=12, seed=None, options=None, render_mode="human", resolution=(224,224), world_path=os.path.join('wrestlers', 'world_data', 'flat.json')):
        self.T = T  # sinusoid period       
        self.blue_wrestler = blue_wrestler
        self.red_wrestler = red_wrestler
        self.blue_voxels = get_voxels(blue_wrestler)
        self.red_voxels = get_voxels(red_wrestler)
        self.blue_colors = [rgb_colors[self.blue_wrestler[i,j]] for i,j,_ in self.blue_voxels]
        self.red_colors = [rgb_colors[self.red_wrestler[i,j]] for i,j,_ in self.red_voxels]
        self.world = EvoWorld.from_json(world_path)
        self.world.add_from_array('Blue', remove_phase(blue_wrestler), 1, 1, connections=None)  # None = all adjacent boxels are connected
        self.world.add_from_array('Red', remove_phase(red_wrestler), 26, 1, connections=None)
        EvoGymBase.__init__(self, self.world, render_mode=render_mode)
        self.blue_action_size = self.get_actuator_indices('Blue').size
        self.red_action_size = self.get_actuator_indices('Red').size
        self.num_actuators =  self.blue_action_size + self.red_action_size
        self.in_phase_actuators = np.concatenate((blue_wrestler[blue_wrestler >= 3] >= 5, red_wrestler[red_wrestler >= 3] >= 5))
        assert len(self.in_phase_actuators) == self.num_actuators
        
        obs, _ = self.reset(seed, options)
        self.action_space = spaces.Box(low= 0, high=1, shape=(0,), dtype=np.float_)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(obs.size,), dtype=np.float_)
        if render_mode in ["screen", "human"]:
            self.default_viewer.set_view_size((30., 18.))
            self.default_viewer.set_pos((16., 9.))
            self.default_viewer.set_resolution((1280, 720))
        else:
            self.default_viewer.set_view_size((30., 30.))
            self.default_viewer.set_pos((16., 1.))
            self.default_viewer.set_resolution(resolution)
        #self.default_viewer.track_objects('ground')

    def get_quadrilaterals(self):
        X, Y = self.object_pos_at_time(self.get_time(), "Blue")
        X, Y = (X-1)/30*224, (Y+14)/30*224
        blue_quads = [np.column_stack([X[voxel], Y[voxel]]) for _, __, voxel in self.blue_voxels]
        X, Y = self.object_pos_at_time(self.get_time(), "Red")
        X, Y = (X-1)/30*224, (Y+14)/30*224
        red_quads = [np.column_stack([X[voxel], Y[voxel]]) for _, __, voxel in self.red_voxels]
        return {"blue": {"quadrilateral": blue_quads, "colors": self.blue_colors}, 
                "red": {"quadrilateral": red_quads, "colors": self.red_colors}}
    
    def get_obs(self):
        obs = np.array([self.get_pos_com_obs("Blue")[0], self.get_pos_com_obs("Red")[0]])
        return obs

    def step(self, action):
        # ignore the action for the moment
        action = np.where(self.in_phase_actuators, (1 + 0.4*np.sin(2*np.pi*self.get_time()/self.T)) * np.ones(self.num_actuators), (1+0.4*np.sin(-2*np.pi*self.get_time()/self.T)) * np.ones(self.num_actuators))
        
        done = super().step({'Blue': action[:self.blue_action_size], 'Red': action[self.blue_action_size:]})
        truncated = False
        if done:
            truncated = True
        return self.get_obs(), 0, done, truncated, {}

    def reset(self, seed=None, options={}):
        super().reset(seed, options)
        info = {}
        return self.get_obs(), info

