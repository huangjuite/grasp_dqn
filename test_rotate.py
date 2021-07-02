
import os
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.replay_buffer import GraspDataset
from utils.visualization import plot_vlaues, visual_dataset
from network import NetworkRotate

torch.random.manual_seed(888)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print(device)
net = NetworkRotate(device).to(device)


replay_buffer = GraspDataset(path='logger.hdf5', normalize=False)
loader = DataLoader(dataset=replay_buffer,
                    batch_size=2,
                    shuffle=True,
                    num_workers=4)


# color, depth, n_color, n_depth, action, reward, done = replay_buffer[random.randint(
#     0, len(replay_buffer)-1)]

# color = torch.unsqueeze(color, 0)
# depth = torch.unsqueeze(depth, 0)


# r_color, r_depth = net(color, depth, math.radians(45.))

# for i in range(r_color.shape[0]):
#     visual_dataset(color[0], depth[0], r_color[i], r_depth[i])
#     plt.show()


for color, depth, n_color, n_depth, action, reward, done in loader:

    color = color.to(device)
    depth = depth.to(device)

    q = net(color, depth).detach()
    print(q.shape)
    # visual_dataset(q[0], depth[0], q[2], depth[0])
    # plt.show()
    break
