
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
from network import NetworkRotate

config = dict(
    gamma=0.9,
    batch_size=1,
    target_update=10,
    epoch=5,
    normalize=False,
)
wandb.init(config=config, project="grasp-ddqn", name='ddqn-rotate-full')
config = wandb.config
print(config)


torch.cuda.empty_cache()
torch.random.manual_seed(777)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
replay_buffer = GraspDataset(path='logger.hdf5', normalize=config.normalize)
train_loader = DataLoader(dataset=replay_buffer,
                          batch_size=config.batch_size,
                          shuffle=True,
                          num_workers=4)


dqn = NetworkRotate(device).to(device)
dqn.train()
dqn_target = NetworkRotate(device).to(device)
dqn_target.load_state_dict(dqn.state_dict())
dqn_target.eval()

criterion = nn.SmoothL1Loss()
optimizer = optim.SGD(dqn.parameters(), lr=2.5e-4,
                       momentum=0.9, weight_decay=2e-5)
step = 0

for e in range(config.epoch):

    for color, depth, n_color, n_depth, action, reward, done in train_loader:

        color = color.to(device)
        depth = depth.to(device)
        n_color = n_color.to(device)
        n_depth = n_depth.to(device)
        b_size = color.shape[0]

        action = action.to(device)
        reward = reward.to(device)
        done = done.to(device)
        indx = torch.linspace(0, b_size-1, b_size,
                              dtype=torch.long, device=device)

        qs = dqn.forward_all_angle(n_color, n_depth).detach()
        indices = []
        for i in range(qs.shape[0]):
            indices.append(torch.nonzero(qs[i] == torch.max(qs[i]))[0])
        next_best_pixel = torch.vstack(indices)

        next_q_value = dqn_target.forward_all_angle(n_color, n_depth).detach()
        next_q_value = next_q_value[
            indx, next_best_pixel[:, 0],
            next_best_pixel[:, 1], next_best_pixel[:, 2]
        ]
        target = reward + config.gamma * next_q_value * (1-done)

        # target_q_value = Variable(torch.zeros(
        #     (b_size, 4, 224, 224), device=device))
        # target_q_value[indx, action[:, 0],
        #                action[:, 1], action[:, 2]] = target

        # action_weight = Variable(torch.zeros(
        #     (b_size, 4, 224, 224), device=device, requires_grad=False))
        # action_weight[indx, action[:, 0], action[:, 1], action[:, 2]] = 1

        curr_q_value = dqn.forward_all_angle(color, depth)[
            indx, action[:, 0], action[:, 1], action[:, 2]]

        loss = criterion(curr_q_value, target)

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
