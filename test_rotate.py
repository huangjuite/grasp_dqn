
import os
import math
import random
import numpy as np
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.replay_buffer import GraspDataset
from utils.visualization import plot_vlaues, visual_dataset
from network import NetworkRotate, PreProcess

torch.random.manual_seed(888)
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print(device)
preprocess = PreProcess()
net = NetworkRotate(device).to(device)


replay_buffer = GraspDataset(path='logger.hdf5', normalize=False)
loader = DataLoader(dataset=replay_buffer,
                    batch_size=1,
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
def preprocessing(color, depth):
    # Zoom 2 times
    color_img_2x = ndimage.zoom(color, zoom=[2, 2, 1], order=0)
    depth_img_2x = ndimage.zoom(depth, zoom=[2, 2],    order=0)
    # Add extra padding to handle rotations inside network
    diag_length = float(color_img_2x.shape[0])*np.sqrt(2)
    diag_length = np.ceil(diag_length/32)*32  # Shrink 32 times in network
    padding_width = int((diag_length - color_img_2x.shape[0])/2)
    print(padding_width)
    # Convert BGR (cv) to RGB
    color_img_2x_b = np.pad(
        color_img_2x[:, :, 0], padding_width, 'constant', constant_values=0)
    color_img_2x_b.shape = (
        color_img_2x_b.shape[0], color_img_2x_b.shape[1], 1)
    color_img_2x_g = np.pad(
        color_img_2x[:, :, 1], padding_width, 'constant', constant_values=0)
    color_img_2x_g.shape = (
        color_img_2x_g.shape[0], color_img_2x_g.shape[1], 1)
    color_img_2x_r = np.pad(
        color_img_2x[:, :, 2], padding_width, 'constant', constant_values=0)
    color_img_2x_r.shape = (
        color_img_2x_r.shape[0], color_img_2x_r.shape[1], 1)
    input_color_img = np.concatenate(
        (color_img_2x_r, color_img_2x_g, color_img_2x_b), axis=2)

    depth_img_2x = np.pad(depth_img_2x, padding_width,
                          'constant', constant_values=0)
    tmp = depth_img_2x.astype(float)
    tmp.shape = (tmp.shape[0], tmp.shape[1], 1)
    input_depth_img = np.concatenate((tmp, tmp, tmp), axis=2)

    print(input_depth_img.shape)
    plt.imshow(input_depth_img[:, :, 0])
    plt.show()

    # Convert to tensor
    # H, W, C - > N, C, H, W
    input_color_img.shape = (
        input_color_img.shape[0], input_color_img.shape[1], input_color_img.shape[2], 1)
    input_depth_img.shape = (
        input_depth_img.shape[0], input_depth_img.shape[1], input_depth_img.shape[2], 1)
    input_color_data = torch.from_numpy(
        input_color_img.astype(np.float32)).permute(3, 2, 0, 1)
    input_depth_data = torch.from_numpy(
        input_depth_img.astype(np.float32)).permute(3, 2, 0, 1)
    return input_color_data, input_depth_data, padding_width


for color, depth, n_color, n_depth, action, reward, done in loader:

    # color = color[0].permute(1, 2, 0).numpy()
    # depth = depth[0][0].numpy()

    # input_color_data, input_depth_data, padding_width = preprocessing(
    #     color, depth)

    # print(input_color_data.shape)
    # print(input_depth_data.shape)

    color, depth = preprocess(color, depth)
    # print(color.shape)
    # print(depth.shape)

    color = color.to(device)
    depth = depth.to(device)

    q = net(color, depth, 0).detach()
    print(q.shape)
    # visual_dataset(q[0], depth[0], q[2], depth[0])
    # plt.show()

    break
