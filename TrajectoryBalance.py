from GNET import TrajectoryBalance
from environment import WaveFunctionCollapseEnvironment,WaveFunctionCollapseEnvironmentWithFeatureMap
from CONVNET import ConvNet,ConvNetFeatureMapsTB
from torch.distributions.categorical import Categorical
import torch
import tqdm
import os
import pickle as pkl
import itertools

vglc_path = "./Super Mario Bros/Processed/"

def get_tiles(processed_levels_path):
    tiles = []
    if "tiles.pkl" not in os.listdir():
        for file_name in os.listdir(processed_levels_path):
            with open (processed_levels_path + file_name,"r") as file:
                processed_level = str(file.read()).split("\n")[:-1]
            for slice in processed_level:
                for tile in slice:
                    tiles.append(tile)
        tiles = list(set(tiles))
        with open("tiles.pkl","wb") as file:
            pkl.dump(tiles,file)
    else:
        with open("tiles.pkl","rb") as file:
            tiles = pkl.load(file)
    return tiles
 
def state_to_tensor(state):
    state = torch.Tensor(state)
    return state

def train_trajectory_balance(FeatureMaps,from_left,seed,full_tile_set,bottom_left):
    if full_tile_set:
        tiles = get_tiles(vglc_path)
    else:
        tiles = ["X","-"]
    num_actions = len(tiles)

    if FeatureMaps:
        forward_model = ConvNetFeatureMapsTB(length,width,num_actions)
        wfc = WaveFunctionCollapseEnvironmentWithFeatureMap(length,width,tiles,epsilon,seed,from_left)
        method_name = "tbfm"
    else:
        forward_model = ConvNet(length,width,(num_actions*2))
        wfc = WaveFunctionCollapseEnvironment(length,width,tiles,epsilon,seed,from_left,bottom_left)
        method_name = "tb"

    if from_left:
        method_name += "lr"
    elif bottom_left:
        method_name += "bl"

    if full_tile_set:
        method_name += "_all_tiles"
    else:
        method_name += "_limited_tile_set"

    F_sa = TrajectoryBalance(forward_model,wfc.backward_policy,num_actions)
    opt = torch.optim.Adam(F_sa.parameters(), 3e-4)
    # opt = torch.optim.SGD(F_sa.parameters(), 3e-4)

    rewards = []
    losses = []
    sampled_levels = []
    minibatch_loss = 0
    update_freq = 1

    logZs = []
    for episode in tqdm.tqdm(range(10000)):
        state, done = wfc.build_empty()
        state = state_to_tensor(state).unsqueeze(0).float()
        P_F_s, P_B_s = F_sa(state)

        total_P_F = 0
        total_P_B = 0
        while not done:
            cat = Categorical(logits=P_F_s)
            action = cat.sample()
            new_state,done = wfc.build_next(state.squeeze(0),action)
            new_state = new_state.unsqueeze(0)
            total_P_F += cat.log_prob(action)

            if done:
                reward = torch.tensor(wfc.get_reward(new_state.squeeze(0))).float()
                rewards.append(reward)
            else:
                reward = 0
            P_F_s, P_B_s = F_sa(state_to_tensor(new_state))
            total_P_B += Categorical(logits=P_B_s).log_prob(action)

            state = new_state

        loss = (F_sa.logZ + total_P_F - torch.log(reward).clip(-20) - total_P_B).pow(2)
        minibatch_loss += loss
        sampled_levels.append(state)
        if episode % update_freq == 0:
            losses.append(minibatch_loss.item())
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            minibatch_loss = 0
            logZs.append(F_sa.logZ.item())
    return sampled_levels,losses,rewards,forward_model,method_name

length = 6
width = 10
epsilon = 0.7

FeatureMaps = False

settings = list(itertools.product([True,False], repeat=4))

os.makedirs("Logs", exist_ok=True)
os.makedirs("Logs/levels_", exist_ok=True)
os.makedirs("Logs/loss_", exist_ok=True)
os.makedirs("Logs/reward_", exist_ok=True)


for from_left,seed,full_tile_set,bottom_left in settings:
    
    sampled_levels,losses,rewards,forward_model,method_name = train_trajectory_balance(FeatureMaps,from_left,seed,full_tile_set,bottom_left)

    with open("Logs/levels_"+method_name+".pkl","wb+") as file:
        pkl.dump(sampled_levels,file)

    with open("Logs/loss_"+method_name+".pkl","wb+") as file:
        pkl.dump(losses,file)

    with open("Logs/reward_"+method_name+".pkl","wb+") as file:
        pkl.dump(rewards,file)

    torch.save(forward_model.state_dict(),"Logs/model_"+method_name+".pt")
