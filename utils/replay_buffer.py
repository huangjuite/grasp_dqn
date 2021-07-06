
import h5py
from typing import Dict
import numpy as np
from torch._C import dtype
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm


class GraspDataset(Dataset):
    def __init__(self, path='../logger.hdf5', normalize=True):

        print("grap dataset: ", path)
        hdf5_path = path
        f = h5py.File(hdf5_path, "r")
        self.size = len(f.keys())
        self.color_buf = np.zeros([self.size, 224, 224, 3], dtype=np.float32)
        self.depth_buf = np.zeros([self.size, 224, 224, 1], dtype=np.float32)
        self.pixel_index_buf = np.zeros([self.size, 3], dtype=np.int)
        self.reward_buf = np.zeros([self.size], dtype=np.float32)
        self.next_color_buf = np.zeros(
            [self.size, 224, 224, 3], dtype=np.float32)
        self.next_depth_buf = np.zeros(
            [self.size, 224, 224, 1], dtype=np.float32)
        self.is_empty_buf = np.zeros([self.size], dtype=np.float32)

        for i, key in enumerate(tqdm(f.keys())):
            group = f[key]
            self.color_buf[i] = group['state/color'][()]/255.0
            self.depth_buf[i] = np.expand_dims(
                group['state/depth'][()], axis=2)/1000.0
            self.pixel_index_buf[i] = group['action'][()]
            self.reward_buf[i] = group['reward'][()]
            self.next_color_buf[i] = group['next_state/color'][()]/255.0
            self.next_depth_buf[i] = np.expand_dims(
                group['next_state/depth'][()], axis=2)/1000.0
            self.is_empty_buf[i] = 1 if group['next_state/empty'][()] else 0

        if normalize:
            print('calculate mean & standard deviation')
            color_mu = np.mean(self.color_buf, axis=(0, 1, 2))
            color_std = np.std(self.color_buf, axis=(0, 1, 2))
            print('color_mu', color_mu)
            print('color_std', color_std)
            depth_mu = np.mean(self.depth_buf, axis=(0, 1, 2))
            depth_std = np.std(self.depth_buf, axis=(0, 1, 2))
            print('depth_mu', depth_mu)
            print('depth_std', depth_std)

            self.color_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=color_mu,
                    std=color_std,
                ),
            ])

            self.depth_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=depth_mu,
                    std=depth_std,
                ),
            ])
        else:
            self.color_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            self.depth_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __getitem__(self, index):
        obs_c = self.color_transform(self.color_buf[index])
        next_obs_c = self.color_transform(self.next_color_buf[index])

        obs_d = self.depth_transform(self.depth_buf[index])
        next_obs_d = self.depth_transform(self.next_depth_buf[index])

        act = self.pixel_index_buf[index]
        rew = self.reward_buf[index]
        done = self.is_empty_buf[index]

        return obs_c, obs_d, next_obs_c, next_obs_d, act, rew, done

    def __len__(self):
        return self.size


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        """Initializate."""
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Store the transition in buffer."""
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size
