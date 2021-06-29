
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.color_feature_encoder = models.densenet121(pretrained=True)

        # self.layers = nn.Sequential(
        #     nn.Linear(in_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, out_dim)
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.color_feature_encoder.features(x)
        return x
