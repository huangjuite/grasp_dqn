#! /usr/bin/env python3
import gym
import math
import os
import random
import numpy as np
import torch
from rainbow.agent import DQNAgent


# environment
env = gym.make("gym_subt:subt-cave-forward-discrete-v0")

seed = 777
np.random.seed(seed)
random.seed(seed)
env.seed(seed)
torch.manual_seed(seed)
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# parameters
num_frames = 2000000
memory_size = 1000
batch_size = 32
target_update = 100

# train
agent = DQNAgent(env, memory_size, batch_size, target_update)
agent.train(num_frames)

# agent.env = gym.wrappers.Monitor(env, "videos", force=True)
# agent.test()
