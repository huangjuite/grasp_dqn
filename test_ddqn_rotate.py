
import os
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.replay_buffer import GraspDataset
from utils.visualization import plot_vlaues
from network import NetworkRotate

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

replay_buffer = GraspDataset(path='logger.hdf5', normalize=False)
loader = DataLoader(dataset=replay_buffer,
                    batch_size=1,
                    shuffle=True,
                    num_workers=4)

dqn = NetworkRotate(device).to(device)
dqn.load_state_dict(torch.load('wandb/latest-run/files/model.pth'))
dqn.eval()


for color, depth, n_color, n_depth, action, reward, done in loader:

    color = color.to(device)
    depth = depth.to(device)

    curr_q_value = dqn(color, depth)[0].detach()
    predict = torch.nonzero(curr_q_value == torch.max(curr_q_value))[0]
    predict = predict.cpu().numpy()
    curr_q_value = curr_q_value.permute(1, 2, 0).cpu().numpy()
    color = color[0].permute(1, 2, 0).cpu().numpy()
    depth = depth[0, 0].cpu().numpy()

    # print(predict)
    plot_vlaues(color, depth, curr_q_value, predict)

    plt.show()
    # break
