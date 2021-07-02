
from functools import reduce
import math
import numpy as np
from numpy.core.numeric import rollaxis

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import ModuleList
from torch.nn import ParameterList
from torch.nn.parameter import Parameter
from torchvision import models


class GraspNet(nn.Module):
    def __init__(self, initialize=True):
        super(GraspNet, self).__init__()

        self.grasp_net = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode="bilinear"),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Upsample(scale_factor=8, mode="bilinear"),
        )

        if initialize:
            for m in self.grasp_net:
                if isinstance(m, nn.Conv2d):
                    nn.init.uniform_(m.weight, -1e-5, 1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.grasp_net(x)


class Network(nn.Module):
    def __init__(self, device, output_head=4):
        super(Network, self).__init__()

        self.color_feature_encoder = models.densenet121(pretrained=True)
        self.depth_feature_encoder = models.densenet121(pretrained=True)

        self.grasp_nets = ModuleList(
            [GraspNet(initialize=False) for i in range(output_head)])

    def forward(self, color: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:

        x_color = self.color_feature_encoder.features(color)

        depth3 = torch.cat((depth, depth, depth), axis=1)
        x_depth = self.depth_feature_encoder.features(depth3)

        x = torch.cat((x_color, x_depth), axis=1)

        y = torch.cat([net(x) for net in self.grasp_nets], axis=1)

        return y


class NetworkRotate(nn.Module):
    def __init__(
        self,
        device,
        rotations=np.array([-90, -45, 0, 45]),
        # rotations=np.array([-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5]),
    ):
        super(NetworkRotate, self).__init__()

        self.color_feature_encoder = models.densenet121(pretrained=True)
        self.depth_feature_encoder = models.densenet121(pretrained=True)

        self.grasp_net = GraspNet(initialize=True)
        self.full_rot_mat = self.get_rot_matrix(rotations).to(device)
        self.full_rot_back_mat = self.get_rot_matrix(-rotations).to(device)
        self.rotate_n = len(rotations)

    def get_rot_matrix(self, degree: np.ndarray) -> torch.Tensor:
        theta = torch.tensor(np.radians(degree))
        c = torch.cos(theta)
        s = torch.sin(theta)
        rot_mat = Variable(torch.zeros(
            theta.shape[0], 2, 3, requires_grad=False))
        rot_mat[:, 0, 0] = c
        rot_mat[:, 0, 1] = -s
        rot_mat[:, 1, 0] = s
        rot_mat[:, 1, 1] = c
        return rot_mat

    def rotate_img(self, x, rot_mat):
        grid = F.affine_grid(rot_mat, x.size())
        x = F.grid_sample(x, grid, mode="nearest")
        return x

    def forward(self, color: torch.Tensor, depth: torch.Tensor, theta: float = None) -> torch.Tensor:
        b_szie = color.shape[0]
        color = torch.repeat_interleave(color, repeats=self.rotate_n, dim=0)
        depth = torch.repeat_interleave(depth, repeats=self.rotate_n, dim=0)
        rot_mat = self.full_rot_mat.repeat(b_szie, 1, 1)
        r_colors = self.rotate_img(color, rot_mat)
        r_depths = self.rotate_img(depth, rot_mat)
        r_depths = r_depths.repeat(1, 3, 1, 1)

        r_feature_color = self.color_feature_encoder.features(r_colors)
        r_feature_depth = self.color_feature_encoder.features(r_depths)
        r_features = torch.cat((r_feature_color, r_feature_depth), dim=1)
        q = self.grasp_net(r_features)
        rot_back_mat = self.full_rot_back_mat.repeat(b_szie, 1, 1)
        q = self.rotate_img(q, rot_back_mat)

        # q = [b0r0, b0r1, b0r2, ... , bnr6, bnr7]
        q = torch.reshape(q, (b_szie, self.rotate_n, 224, 224))

        return q
