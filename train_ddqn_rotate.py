
import os
import sys
import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils.replay_buffer import GraspDataset
from network import NetworkRotate, PreProcess

config = dict(
    gamma=0.5,
    target_update=50,
    epoch=4,
    normalize=True,
)
wandb.init(config=config, project="grasp-ddqn", name='ddqn-rotate-4')
config = wandb.config
print(config)


torch.cuda.empty_cache()
torch.random.manual_seed(777)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
replay_buffer = GraspDataset(path='logger.hdf5', normalize=config.normalize)
train_loader = DataLoader(dataset=replay_buffer,
                          batch_size=1,
                          shuffle=True,
                          num_workers=4)

rotations = np.array([-90, -45, 0, 45])
# rotations=np.array([-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5])

preprocess = PreProcess()
dqn = NetworkRotate(device, rotations).to(device)
dqn_target = NetworkRotate(device, rotations).to(device)
dqn_target.load_state_dict(dqn.state_dict())
dqn_target.eval()

criterion = nn.SmoothL1Loss()
optimizer = optim.SGD(
    [
        {'params': dqn.grasp_net.parameters(), 'lr': 2.5e-4},
        {'params': dqn.color_feature_encoder.parameters(), 'lr': 5e-5},
        {'params': dqn.depth_feature_encoder.parameters(), 'lr': 5e-5},
    ],
    lr=2.5e-4,
    momentum=0.9,
    weight_decay=2e-5,
)
step = 0

for e in range(config.epoch):

    for color, depth, n_color, n_depth, action, reward, done in train_loader:
        color, depth = preprocess(color, depth)
        n_color, n_depth = preprocess(n_color, n_depth)

        color = color.to(device)
        depth = depth.to(device)
        n_color = n_color.to(device)
        n_depth = n_depth.to(device)
        b_size = color.shape[0]

        action = action.to(device)[0]
        reward = reward.to(device)[0]
        done = done.to(device)[0]

        # find best action using behavior net
        qs = torch.zeros((len(rotations), 224, 224))
        for theta_idx in range(len(rotations)):
            qs[theta_idx] = dqn(n_color, n_depth, theta_idx).detach()
        best_action = torch.nonzero(qs == torch.max(qs))[0]

        # calculate best action value in target net
        next_qs = dqn_target(n_color, n_depth, best_action[0])
        next_q_value = next_qs[0, best_action[1], best_action[2]]
        target = reward + config.gamma * next_q_value * (1-done)

        # make target value label
        target_q_value = Variable(torch.zeros(
            (1, 224, 224), device=device))
        target_q_value[0, action[1], action[2]] = target
        action_weight = Variable(torch.zeros(
            (1, 224, 224), device=device, requires_grad=False))
        action_weight[0, action[1], action[2]] = 1

        # get current q value
        curr_q_value = dqn(color, depth, action[0])[0, action[1], action[2]]

        # calculate TD error
        loss = criterion(curr_q_value, target_q_value)*action_weight
        loss = loss.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({'TD_error': loss.item()})

        step += 1

        # hard update is needed
        if step % config.target_update == 0:
            print('hard update target net, step: ', step)
            dqn_target.load_state_dict(dqn.state_dict())

torch.save(dqn.state_dict(), os.path.join(wandb.run.dir, "model.pth"))
