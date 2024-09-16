import numpy as np
import random
import torch
from Utility.Mario import Fitness
from Utility.GridTools import rows_into_columns

class WaveFunctionCollapseEnvironment():
    def __init__(self,length,width,tiles,epsilon,with_seed = False, from_left = False, bottom_left = False):
        self.tiles = tiles
        self.with_seed = with_seed
        self.from_left = from_left
        self.bottom_left = bottom_left

        # Unknown Tile
        self.tiles.append("U")
        # Next Tile
        self.tiles.append("N")
        
        self.length = length
        self.width = width
        self.num_actions = len(tiles) 
        self.tiles_map = {tile:idx for idx,tile in enumerate(self.tiles)}
        self.ids_map = {idx:tile for tile,idx in self.tiles_map.items()}
        self.epsilon = epsilon

    def is_done(self,state):
        return self.tiles_map['U'] not in state


    def is_border(self, state, x, y):
        # Assuming state is a NumPy array
        u_tile = self.tiles_map["U"]
        # Check all surrounding tiles at once
        checks = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        return any((0 <= nx < self.length and 0 <= ny < self.width and state[nx, ny] == u_tile) for nx, ny in checks)

    def backward_policy(self,state:torch.Tensor):
        parents = []
        parents_actions = []
        state_temp = state.clone().numpy()
        current_state_N = np.where(state_temp == self.tiles_map["N"])
        state_temp[current_state_N] = self.tiles_map["U"]
        for x in range(self.length):
            for y in range(self.width): 
                # Replace this with np.where
                if state_temp[x,y] != self.tiles_map["U"] and self.is_border(state_temp,x,y):
                    parent_state = state_temp.copy()
                    parents_actions.append(parent_state[x,y])
                    parent_state[x,y] = self.tiles_map['N']
                    parents.append(parent_state)
        parents = [torch.from_numpy(parent) for parent in parents]

        # for the last step, when we have built the whole level
        if not parents:
            for x in range(self.length):
                for y in range(self.width):
                        if random.random() > self.epsilon: 
                            parent_state = state_temp.copy()
                            parents_actions.append(parent_state[x,y])
                            parent_state[x,y] = self.tiles_map['N']
                            parents.append(parent_state)
        return parents,parents_actions

    def find_next(self, state):
        # Convert state to NumPy array for faster operations
        state_np = state.numpy()
        u_tile = self.tiles_map["U"]
        
        # Initialize an empty list to collect possible next positions
        possible = []
        
        # Only iterate through border tiles that are not 'U'
        borders = np.argwhere(state_np != u_tile)
        if self.from_left:
            for x, y in borders:
                if y < self.width - 1 and state_np[x, y + 1] == u_tile:
                    possible.append((x, y + 1))
        elif self.bottom_left:
            for x, y in borders:
                checks = [(x + 1, y), (x, y + 1)]
                possible.extend((nx, ny) for nx, ny in checks if 0 <= nx < self.length and 0 <= ny < self.width and state_np[nx, ny] == u_tile)
        else:
            for x, y in borders:
                checks = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                possible.extend((nx, ny) for nx, ny in checks if 0 <= nx < self.length and 0 <= ny < self.width and state_np[nx, ny] == u_tile)
    
        # Select a random next position if possible is not empty
        # if possible:
        return random.choice(possible)
        # else:
        #     return None  # Or handle this case as you see fit

    def seed_level(self,level):

        # for x in range(self.length):
        level[:,0] = self.tiles_map["-"]
        level[:,self.width - 1] = self.tiles_map["-"] 

        # for y in range(self.width):
            # level[0,y] = self.tiles_map["-"]
        level[self.length - 1,:] = self.tiles_map["X"] 

        return level
    
    def build_empty(self):

        state = np.full((self.length,self.width),self.tiles_map['U'])
        state = torch.from_numpy(state)

        if self.with_seed:
            state = self.seed_level(state)

        if self.from_left:
            x = random.randrange(1,self.length-2)
            y = 0
        elif self.bottom_left:
            x = self.length-2
            y = 1
        else:
            x = random.randrange(1,self.length-2)
            y = random.randrange(1,self.width-2)
        state[x,y] = self.tiles_map['N']

        return state, False

    def build_next(self,state,action):
        current_state_N = np.where(state == self.tiles_map["N"])

        state[current_state_N] = action.float()
        
        done = True
        if not self.is_done(state):
            tile = self.find_next(state)
            state[tile] = self.tiles_map['N']
            done = False

        return state, done
    

    def array_to_string(self, state):
        # Using a list comprehension and map for efficient conversion
        return ["".join(map(self.ids_map.get, row)) for row in state.tolist()]


    def get_reward(self,state):
        level = state.clone().detach().numpy()
        level = self.array_to_string(level)
        level = rows_into_columns(level)
        return Fitness.percent_playable(level,self.length)

class WaveFunctionCollapseEnvironmentWithFeatureMap():
    def __init__(self,length,width,tiles,epsilon,with_seed = False, from_left = False, bottom_left = False):
        self.length = length
        self.width = width
        
        self.tiles = tiles
        self.with_seed = with_seed
        self.from_left = from_left
        self.bottom_left = bottom_left

        # Unknown Tile
        self.tiles.append("U")
        # Next Tile
        self.tiles.append("N")
        
        self.num_actions = len(tiles) 
        
        self.tiles_map = {tile:idx for idx,tile in enumerate(self.tiles)}
        self.ids_map = {idx:tile for tile,idx in self.tiles_map.items()}
        
        self.epsilon = epsilon

    def feature_map_to_level(self,feature_layers):
            
        level_array = np.zeros((self.length, self.width), dtype=int)
        for i in range(self.length):
            for j in range(self.width):
                tile_indices = np.where(feature_layers[:, i, j] == 1)[0]
                if len(tile_indices) > 0:
                    level_array[i, j] = np.max(tile_indices)
        
        return level_array

    def level_to_feature_map(self,level):
        feature_layers = np.zeros((self.num_actions, self.length, self.width), dtype=int)
            
        for tile in self.tiles:
                feature_layers[self.tiles_map[tile]][level == self.tiles_map[tile]] = 1
            
        return feature_layers

    def is_done(self,state):
        return 1 not in state[self.tiles_map['U']]

    def surrounding(self,state,x,y):
        possible = []
        if x < self.length - 1:
            if state[self.tiles_map["U"]][x+1,y] == 1:
                possible.append((x+1,y))
        if x > 0:
            if state[self.tiles_map["U"]][x-1,y] == 1:
                possible.append((x-1,y))
        if y < self.width - 1:
            if state[self.tiles_map["U"]][x,y+1] == 1:
                possible.append((x,y+1))
        if y > 0:
            if state[self.tiles_map["U"]][x,y-1] == 1:
                possible.append((x,y-1))
        return possible
    
    def is_border(self,state,x,y):

        is_border = False
        if x < self.length - 1:
            if state[x+1,y] == self.tiles_map["U"]:
                is_border = True
        if x > 0:
            if state[x-1,y] == self.tiles_map["U"]:
                is_border = True
        if y < self.width - 1:
            if state[x,y+1] == self.tiles_map["U"]:
                is_border = True
        if y > 0:
            if state[x,y-1] == self.tiles_map["U"]:
                is_border = True

        return is_border

    def backward_policy(self,state:torch.Tensor):
        parents = []
        parents_actions = []
        state_temp = state.clone().numpy()
        state_temp = self.feature_map_to_level(state_temp)
        current_state_N = np.where(state_temp == self.tiles_map["N"])
        state_temp[current_state_N] = self.tiles_map["U"]
        for x in range(self.length):
            for y in range(self.width):
                if state_temp[x,y] != self.tiles_map["U"] and self.is_border(state_temp,x,y):
                    parent_state = state_temp.copy()
                    parents_actions.append(parent_state[x,y])
                    parent_state[x,y] = self.tiles_map['N']
                    parents.append(self.level_to_feature_map(parent_state))
        parents = [torch.from_numpy(parent) for parent in parents]
        if not parents:
            for x in range(self.length):
                for y in range(self.width):
                        # if random.random() > self.epsilon: 
                        parent_state = state_temp.copy()
                        parents_actions.append(parent_state[x,y])
                        parent_state[x,y] = self.tiles_map['N']
                        parents.append(self.level_to_feature_map(parent_state))
        return parents,parents_actions


    def find_next(self,state):
        possible = []
        for x in range(self.length):
            for y in range(self.width):
                if state[self.tiles_map["U"]][x,y] != 1:
                    possible.extend(self.surrounding(state,x,y))
        return random.choice(possible)

    def seed_level(self,level):

        for x in range(self.length):
            level[x,0] = self.tiles_map["-"]
            level[x,self.width - 1] = self.tiles_map["-"] 

        for y in range(self.width):
            level[self.length - 1,y] = self.tiles_map["X"] 

        return level

    def build_empty(self):
        state = np.full((self.length,self.width),self.tiles_map['U'])
        x = random. randrange(1,self.length)
        y = random. randrange(1,self.width)
        state[x,y] = self.tiles_map['N']

        if self.with_seed:
            state = self.seed_level(state)

        state = self.level_to_feature_map(state)
        state = torch.from_numpy(state)

        return state, False

    def build_next(self,state,action):
        current_state_N = np.where(state[self.tiles_map["N"]] == 1)
        a = action.numpy()[0]
        state[self.tiles_map["N"]][current_state_N] = 0
        state[a][current_state_N] = 1
        done = True
        if not self.is_done(state):
            tile = self.find_next(state)
            state[self.tiles_map["U"]][tile] = 0
            state[self.tiles_map["N"]][tile] = 1
            done = False

        return state, done
    
    def array_to_string(self,state):
        tiles = state.copy().tolist()
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                tiles[i][j] = self.ids_map[state[i, j]]
        
        level = []
        for i in range(state.shape[0]):
            level.append("".join(tiles[i]))
        return level


    def get_reward(self,state):
        level = state.clone().detach().numpy()
        level = self.feature_map_to_level(level)
        level = self.array_to_string(level)
        level = rows_into_columns(level)
        return Fitness.percent_playable(level,self.length)
