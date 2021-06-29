
import random
import numpy as np
import torch
from replay_buffer import GraspDataset
from matplotlib import pyplot as plt


replay_buffer = GraspDataset(path='../logger.hdf5', normalize=True)

for i in range(10):
    color, depth, n_color, n_depth, action, reward, done = replay_buffer[random.randint(0, len(replay_buffer)-1)]

    print('---------------')
    print('action', action)
    print('reward', reward)
    print('done', done)


    f, ax = plt.subplots(2, 2)

    ax[0,0].imshow(color.permute(1, 2, 0))
    ax[0,1].imshow(torch.squeeze(depth))
    ax[1,0].imshow(n_color.permute(1, 2, 0))
    ax[1,1].imshow(torch.squeeze(n_depth))
    plt.show()
