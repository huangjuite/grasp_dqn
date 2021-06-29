
import torch
import random
from torch.utils.data import DataLoader


from network import Network
from replay_buffer import GraspDataset

replay_buffer = GraspDataset(path='../logger.hdf5', normalize=True)
train_loader = DataLoader(dataset=replay_buffer,
                          batch_size=8,
                          shuffle=True,
                          num_workers=4)

color, depth, n_color, n_depth, action, reward, done = replay_buffer[random.randint(
    0, len(replay_buffer)-1)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Network().to(device)

for color, depth, n_color, n_depth, action, reward, done in train_loader:
    color = color.to(device)
    x = net(color)
    print(x.shape)
    break