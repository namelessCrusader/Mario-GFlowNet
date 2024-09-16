from GNET import FlowMatching
from environment import WaveFunctionCollapseEnvironment,WaveFunctionCollapseEnvironmentWithFeatureMap
from CONVNET import ConvNet,ConvNetFeatureMaps
from torch.distributions.categorical import Categorical
import torch
import numpy as np
import tqdm
import os
import pickle as pkl
import itertools

epsilon = 0.001
alpha = 2
beta = 1
eta = 1

vglc_path = "/home/nisargparikh/Desktop/CS 7170/Final/VGLC/TheVGLC-master/Super Mario Bros/Processed/"

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
    state = torch.Tensor(state.tolist())
    return state
 

def train_flow_matching(FeatureMaps,from_left,Stable,seed,full_tile_set, bottom_left): 
    if full_tile_set:
        tiles = get_tiles(vglc_path)
    else:
        tiles = ["X","-"]
    num_actions = len(tiles)


    if FeatureMaps:
        forward_model = ConvNetFeatureMaps(length,width,num_actions)
        #Not fully implemented yet, from left, bottom left remaining
        wfc = WaveFunctionCollapseEnvironmentWithFeatureMap(length,width,tiles,epsilon,seed)
        method_name = "fmfm"
    else:
        forward_model = ConvNet(length,width,num_actions)
        wfc = WaveFunctionCollapseEnvironment(length,width,tiles,epsilon,seed,from_left,bottom_left)
        method_name = "fm"

    if Stable:
        method_name = "s" + method_name

    if from_left:
        method_name += "lr"
    elif bottom_left:
        method_name += "bl"
    if full_tile_set:
        method_name += "_all_tiles"
    else:
        method_name += "_limited_tile_set"

    F_sa = FlowMatching(forward_model,wfc.backward_policy,num_actions + 2)
    opt = torch.optim.Adam(F_sa.parameters(), 3e-4)

    losses = []
    sampled_levels = []
    rewards = []

    minibatch_loss = 0
    update_freq = 1

    for episode in tqdm.tqdm(range(num_episodes)):
        state, done = wfc.build_empty()
        state = state_to_tensor(state).unsqueeze(0)
        edge_flow_prediction = F_sa(state)
        minibatch_loss = 0
        while not done:
            policy = edge_flow_prediction / edge_flow_prediction.sum()
            action = Categorical(probs=policy).sample()

            new_state, done = wfc.build_next(state.squeeze(0),action)
            parent_states, parent_actions = F_sa.backward(new_state)

            px = torch.stack([state_to_tensor(state) for state in parent_states])
            pa = torch.tensor(parent_actions).long()

            parent_edge_flow_preds = F_sa(px)[torch.arange(len(parent_states)), pa]
            
            if done:
                reward = wfc.get_reward(new_state)
                rewards.append(reward)
                edge_flow_prediction = torch.zeros(num_actions)
            else:
                reward = 0
                new_state = state_to_tensor(state)
                edge_flow_prediction = F_sa(new_state)
                
            parent_flow = torch.as_tensor(parent_edge_flow_preds.sum())
            flow = edge_flow_prediction.sum() + reward

            if Stable:
                abs_flow = torch.log(1 + 0.01 * torch.pow(torch.abs(parent_flow - flow),alpha))
                side_loss = torch.pow(1 + eta * (flow + parent_flow), beta)
                flow_mismatch = abs_flow * side_loss   
            else:
                # flow_mismatch = (torch.log(parent_flow) - torch.log(flow)).pow(2)
                flow_mismatch = (torch.logsumexp(parent_flow,0) - torch.logsumexp(flow,0)).pow(2)
                
            minibatch_loss += flow_mismatch  # Accumulate
            state = new_state

        sampled_levels.append(state)
        if episode % update_freq == 0:
            losses.append(minibatch_loss.item())
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            minibatch_loss = 0
    return sampled_levels,losses,rewards,forward_model, method_name

# FeatureMaps = False
# from_left = True
# Stable = False
# seed = True
# full_tile_set = False

length = 10
width = 10
epsilon = 0.7

num_episodes = 5000

# REDO SFMLR_ALL_TILES once more later, one of the runs showed it getting better 
settings = list(itertools.product([True,False], repeat=3))

for from_left,Stable,full_tile_set in settings:

    seed = True
    FeatureMaps = False
    bottom_left = False
    
    sampled_levels,losses,rewards,forward_model,method_name = train_flow_matching(FeatureMaps,from_left,Stable,seed,full_tile_set, bottom_left)

    with open("Logs/levels_"+method_name+".pkl","wb+") as file:
        pkl.dump(sampled_levels,file)

    with open("Logs/loss_"+method_name+".pkl","wb+") as file:
        pkl.dump(losses,file)

    with open("Logs/reward_"+method_name+".pkl","wb+") as file:
        pkl.dump(rewards,file)

    torch.save(forward_model.state_dict(),"Logs/model_"+method_name+".pt")

settings = list(itertools.product([True,False], repeat=2))

for Stable,full_tile_set in settings:

    from_left = False
    seed = True
    FeatureMaps = False
    bottom_left = True
    
    sampled_levels,losses,rewards,forward_model,method_name = train_flow_matching(FeatureMaps,from_left,Stable,seed,full_tile_set, bottom_left)

    with open("Logs/levels_"+method_name+".pkl","wb+") as file:
        pkl.dump(sampled_levels,file)

    with open("Logs/loss_"+method_name+".pkl","wb+") as file:
        pkl.dump(losses,file)

    with open("Logs/reward_"+method_name+".pkl","wb+") as file:
        pkl.dump(rewards,file)

    torch.save(forward_model.state_dict(),"Logs/model_"+method_name+".pt")

settings = list(itertools.product([True,False], repeat=2))

for Stable,full_tile_set in settings:

    from_left = False
    seed = True
    FeatureMaps = True
    bottom_left = False
    
    sampled_levels,losses,rewards,forward_model,method_name = train_flow_matching(FeatureMaps,from_left,Stable,seed,full_tile_set, bottom_left)

    with open("Logs/levels_"+method_name+".pkl","wb+") as file:
        pkl.dump(sampled_levels,file)

    with open("Logs/loss_"+method_name+".pkl","wb+") as file:
        pkl.dump(losses,file)

    with open("Logs/reward_"+method_name+".pkl","wb+") as file:
        pkl.dump(rewards,file)

    torch.save(forward_model.state_dict(),"Logs/model_"+method_name+".pt")
