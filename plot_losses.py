import pickle as pkl
import matplotlib.pyplot as plt
import plot
import numpy as np
import os

loss_dir = "./Logs/Losses/"
rewards_dir = "./Logs/Rewards/" 
plots_dir = "./Plots/"
for name in os.listdir(loss_dir):
    name = name.split(".")[0]
    with open(loss_dir+name+".pkl","rb") as file:
        losses = pkl.load(file)

    plt.figure(figsize=(10,3))
    plt.plot(plot.moving_average(np.array(losses)))

    plt.yscale('log')
    plt.savefig(plots_dir+name+".png")

    plt.clf()

