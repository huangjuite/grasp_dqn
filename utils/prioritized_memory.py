
import torch
import numpy as np
import random
import pickle
from .SumTree import SumTree
from collections import namedtuple
import h5py

# Revised from: https://github.com/rlcode/per and https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py


class Memory:
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment = 0.001
    abs_err_upper = 1.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def add(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment])

        for i in range(n):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        sampling_probabilities = priorities / self.tree.total  # (1)
        is_weight = np.power(self.tree.n_entries *
                             sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()  # 3.4

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        #print("Data index {} priority set from {} to {}".format(idx - self.tree.capacity + 1, self.tree.tree[idx], p))
        self.tree.update(idx, p)

    def save_memory(self, root_path, name):
        f = open(root_path+name, 'wb')
        pickle.dump(self.tree, f)
        f.close()

    def load_memory(self, file_path):
        with open(file_path, 'rb') as file:
            self.tree = pickle.load(file)
            print("Loaded {} data".format(self.tree.length))

    @property
    def length(self):
        return self.tree.n_entries


Transition = namedtuple(
    'Transition',
    ['color', 'depth', 'pixel_idx', 'reward',
        'next_color', 'next_depth', 'is_empty']
)


class ReplayBuffer():

    def __init__(self, hdf5_path='logger05.hdf5') -> None:

        f = h5py.File(hdf5_path, "r")
        memory_size = len(f.keys())

        self.gripper_memory = Memory(memory_size)

        for key in f.keys():
            group = f[key]
            color = group['state/color'][()].astype(np.float32)/255.0
            depth = group['state/depth'][()].astype(np.float32)/1000.0
            pixel_index = group['action'][()].astype(np.int)
            reward = group['reward'][()]
            next_color = group['next_state/color'][()].astype(np.float32)/255.0
            next_depth = group['next_state/depth'][()].astype(np.float32)/1000.0
            is_empty = 1 if group['next_state/empty'][()] else 0

            transition = Transition(
                color, depth, pixel_index, reward, next_color, next_depth, is_empty)
            self.gripper_memory.add(transition)

        print("Gripper_Buffer: {}".format(self.gripper_memory.length))

    def sample_data(self, batch_size):
        done = False
        mini_batch = []
        idxs = []
        is_weight = []
        while not done:
            success = True
            mini_batch, idxs, is_weight = self.gripper_memory.sample(batch_size)
            for transition in mini_batch:
                success = success and isinstance(transition, Transition)
            if success:
                done = True
        return mini_batch, idxs, is_weight



def to_torch(mini_batch: Transition, device: torch.device):
    color = torch.tensor(mini_batch.color).permute(2, 0, 1).view(
        1, 3, 224, 224).to(device).to(device)

    depth = torch.tensor(mini_batch.depth).view(1, 1, 224, 224).to(device)

    pixel_index = torch.tensor(mini_batch.pixel_idx).to(device)

    n_color = torch.tensor(mini_batch.next_color).permute(
        2, 0, 1).view(1, 3, 224, 224).to(device)

    n_depth = torch.tensor(mini_batch.next_depth).view(
        1, 1, 224, 224).to(device)

    reward = torch.tensor(mini_batch.reward).to(device)
    done = torch.tensor(mini_batch.is_empty).to(device)

    return color, depth, n_color, n_depth, pixel_index, reward, done