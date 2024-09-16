from PIL import Image
import numpy as np
import pickle
import random
import os

# Mapping between images to Tiles
# Mapping between ASCII characters and image filenames
ascii_to_image = {
    "X": "floor.png",
    "-": "sky.png",
    "S" : "brick_floor_breakable.png",
    "?" : "question.png",
    "Q" : "spent_question.png",
    "E" : "mushroom_enemy.png",
    "<" : "pipe_top_left.png",
    ">" : "pipe_top_right.png",
    "[" : "pipe_left.png",
    "]" : "pipe_right.png",
    "o" : "coin.png",
    "B" : "cannon_top.png",
    "b" : "cannon_bottom.png"
}

# Load images
image_mapping = {char: Image.open("./tileset/" + filename) for char, filename in ascii_to_image.items()}

def generate_image_from_ascii(ascii_level, tile_size):
    width = len(ascii_level[0]) * tile_size
    height = len(ascii_level) * tile_size
    composite_image = Image.new('RGB', (width, height), (255, 255, 255))  # Create a blank white canvas

    for y, row in enumerate(ascii_level):
        for x, char in enumerate(row):
            if char in image_mapping:
                tile_image = image_mapping[char]
                composite_image.paste(tile_image, (x * tile_size, y * tile_size))

    return composite_image

def feature_map_to_level(feature_layers,length,width):
        
    level_array = np.zeros((length, width), dtype=int)
    for i in range(length):
        for j in range(width):
            tile_indices = np.where(feature_layers[:, i, j] == 1)[0]
            if len(tile_indices) > 0:
                level_array[i, j] = np.max(tile_indices)

    return level_array

levels_dir = "./Logs/Levels/"
rewards_dir = "./Logs/Rewards/"

for name in os.listdir(levels_dir):
    if "all_tiles" in name:
        embedding_to_tile = {0: '[', 1: '?', 2: '<', 3: 'X', 4: '-', 5: ']', 6: 'E', 7: '>', 8: 'Q', 9: 'o', 10: 'S', 11: 'b', 12: 'B', 13: 'U', 14: 'N'}
    else:
        embedding_to_tile = {0: 'X', 1: '-', 2: 'U', 3: 'N'}

    name = name.split(".")[0]

    if name not in os.listdir("./Levels"):
        os.mkdir("./Levels/"+name)
        os.mkdir("./Levels/"+name+"/Playable")
        os.mkdir("./Levels/"+name+"/Unplayable")

        with open(levels_dir+name+".pkl","rb") as file:
            levels = pickle.load(file)

        reward_name = "reward_"+"_".join(name.split("_")[1:])
        with open(rewards_dir+reward_name+".pkl","rb") as file:
            rewards = pickle.load(file)

        for i in range(len(rewards)):
            level = levels[i]

            level = level.clone().numpy().tolist()
            if isinstance(level[0][0],list):
                level = level[0]
            if "fmfm" in name:
                try:
                    level = feature_map_to_level(np.array(level),len(level[0]),len(level[1])).tolist()
                except:
                    input(name)
            # try:
            ascii_level = level.copy()
            # except:
            #     ascii_level = level.copy().tolist()
            try:
                for y, row in enumerate(level):
                    for x, char in enumerate(row):
                        try:
                            ascii_level[y][x] = embedding_to_tile[char]
                        except:
                            ascii_level[y][x] = char
                            if "fmfm" in name:
                                ascii_level[y][x] = embedding_to_tile[char]


                tile_size = 16  # Adjust this based on your preference and image sizes
                result_image = generate_image_from_ascii(ascii_level, tile_size)
            except:
                input(name)
            if rewards[i] == 1:
                result_image.save("./Levels/"+name+"/Playable/"+str(i)+".png") 
            else:
                result_image.save("./Levels/"+name+"/Unplayable/"+str(i)+".png") 
            
