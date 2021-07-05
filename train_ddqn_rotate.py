
import os
import sys
import wandb
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils.prioritized_memory import ReplayBuffer, to_torch
from network import NetworkRotate, PreProcess

config = dict(
    gamma=0.9,
    iteration=1000,
    target_update=10,
    batch_size=10,
    normalize=True,
)
wandb.init(config=config, project="grasp-ddqn", name='ddqn-rotate-4')
config = wandb.config
print(config)


torch.cuda.empty_cache()
torch.random.manual_seed(777)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

rotations = np.array([-90, -45, 0, 45])
# rotations=np.array([-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5])

preprocess = PreProcess()
dqn = NetworkRotate(device, rotations).to(device)
dqn_target = NetworkRotate(device, rotations).to(device)
dqn_target.load_state_dict(dqn.state_dict())

criterion = nn.SmoothL1Loss(reduce=False)
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

replay_buffer = ReplayBuffer('logger05.hdf5')


def get_target_value(n_color, n_depth, reward, done):
    # find best action using behavior net
    qs = torch.zeros((len(rotations), 224, 224))
    for theta_idx in range(len(rotations)):
        qs[theta_idx] = dqn(n_color, n_depth, theta_idx).detach()
    best_action = torch.nonzero(qs == torch.max(qs))[0]

    # calculate best action value in target net
    next_qs = dqn_target(n_color, n_depth, best_action[0]).detach()
    next_q_value = next_qs[0, best_action[1], best_action[2]]
    target = reward + config.gamma * next_q_value * (1-done)
    return target


for iteration in range(config.iteration):

    mini_batch, idxs, is_weight, old_q, loss_list = [], [], [], [], []

    _mini_batch, _idxs, _is_weight = replay_buffer.sample_data(
        config.batch_size)
    mini_batch += _mini_batch
    idxs += _idxs
    is_weight += list(_is_weight)

    optimizer.zero_grad()

    for j in range(len(mini_batch)):
        color, depth, n_color, n_depth, action, reward, done = to_torch(
            mini_batch[j], device)

        color, depth = preprocess(color, depth)
        n_color, n_depth = preprocess(n_color, n_depth)

        target = get_target_value(n_color, n_depth, reward, done).detach()

        # make target value label
        target_q_value = torch.zeros(
            (1, 224, 224), device=device, requires_grad=False)
        target_q_value[0, action[1], action[2]] = target
        action_weight = torch.zeros(
            (1, 224, 224), device=device, requires_grad=False)
        action_weight[0, action[1], action[2]] = 1

        # get current q value
        curr_q_value = dqn(color, depth, action[0])

        # calculate TD error
        loss = criterion(curr_q_value, target_q_value)*action_weight
        loss = loss.sum()*is_weight[j]/config.batch_size
        loss.backward()

        # train symetric
        sym_q_value = dqn(color, depth, action[0]+len(rotations))

        # calculate TD error
        loss_sym = criterion(sym_q_value, target_q_value)*action_weight
        loss_sym = loss_sym.sum()*is_weight[j]/config.batch_size
        loss_sym.backward()

        loss_list.append((loss_sym.item()+loss.item())/2.)

    optimizer.step()

    # update priority
    for j in range(len(mini_batch)):
        color, depth, n_color, n_depth, action, reward, done = to_torch(
            mini_batch[j], device)

        color, depth = preprocess(color, depth)
        n_color, n_depth = preprocess(n_color, n_depth)

        td_target = get_target_value(
            n_color, n_depth, reward, done).detach().cpu().numpy()
        new_value = dqn(color, depth, action[0])[
            0, action[1], action[2]].detach().cpu().numpy()
        replay_buffer.gripper_memory.update(idxs[j], td_target-new_value)

    # log
    print('iteration:%d, loss:%f' % (iteration, np.sum(loss_list)))
    wandb.log({'TD_error': np.sum(loss_list)})

    # hard update is needed
    if iteration % config.target_update == 0:
        print('hard update target net, iteration: ', iteration)
        dqn_target.load_state_dict(dqn.state_dict())

torch.save(dqn.state_dict(), os.path.join(wandb.run.dir, "model.pth"))
