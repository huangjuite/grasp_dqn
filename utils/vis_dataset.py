
import random
import numpy as np
import torch
from replay_buffer import GraspDataset
from matplotlib import pyplot as plt

from visualization import visual_dataset

replay_buffer = GraspDataset(path='../logger.hdf5', normalize=True)

for i in range(10):
    color, depth, n_color, n_depth, action, reward, done = replay_buffer[random.randint(
        0, len(replay_buffer)-1)]

    print('---------------')
    print('action', action)
    print('reward', reward)
    print('done', done)

    visual_dataset(color, depth, n_color, n_depth)
    plt.show()

