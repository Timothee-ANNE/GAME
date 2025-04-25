# %% Imports
from flax.struct import dataclass
import chex


# %% Dataclasses
@dataclass
class Terrain:
    building: chex.Array
    water: chex.Array
    forest: chex.Array
    basemap: chex.Array
    
    def __getitem__(self, index):  # to allow slicing operations
        return Terrain(
            building=self.building[index],
            water=self.water[index],
            forest=self.forest[index],
            basemap=self.basemap[index],
        )

# %%
