
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
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
