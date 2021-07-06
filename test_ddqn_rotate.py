
import os
import numpy
import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.replay_buffer import GraspDataset
from utils.visualization import plot_vlaues
from network import NetworkRotate, PreProcess

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

replay_buffer = GraspDataset(path='Logger05_8.hdf5', normalize=False)
loader = DataLoader(dataset=replay_buffer,
                    batch_size=1,
                    shuffle=True,
                    num_workers=4)

# rotations = np.array([-90, -45, 0, 45])
rotations=np.array([-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5])

preprocess = PreProcess()
dqn = NetworkRotate(device, rotations).to(device)
dqn.load_state_dict(torch.load('model_0706_8.pth'))


for color, depth, n_color, n_depth, action, reward, done in loader:

    pcolor, pdepth = preprocess(color, depth)

    pcolor = pcolor.to(device)
    pdepth = pdepth.to(device)

    qs = torch.zeros(len(rotations), 224, 224)
    for theta_idx in range(len(rotations)):
        qs[theta_idx] = dqn(pcolor, pdepth, theta_idx).detach()
    best_action = torch.nonzero(qs == torch.max(qs))[0].cpu().numpy()
    qs = qs.permute(1, 2, 0).cpu().numpy()
    color = color[0].permute(1, 2, 0).numpy()
    depth = depth[0, 0].numpy()

    print('action ', action[0])
    print('selection ', best_action)
    plot_vlaues(color, depth, qs, best_action)

    plt.show()
    # break
