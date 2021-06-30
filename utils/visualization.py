
import numpy as np
from numpy import ma
import torch
import matplotlib.pyplot as plt
import cv2


def visual_dataset(color: torch.Tensor, depth: torch.Tensor,
                   n_color: torch.Tensor, n_depth: torch.Tensor) -> None:
    f, ax = plt.subplots(2, 2, figsize=(10, 10))

    ax[0, 0].imshow(color.permute(1, 2, 0))
    ax[0, 1].imshow(torch.squeeze(depth))
    ax[1, 0].imshow(n_color.permute(1, 2, 0))
    ax[1, 1].imshow(torch.squeeze(n_depth))


def to_colorMap(q_values: np.ndarray) -> np.ndarray:
    maxq = np.max(q_values)
    minq = np.min(q_values)
    q_values = 1-(q_values-minq)/(maxq-minq)
    q_values = (q_values*255).astype(np.uint8)
    heatmaps = []
    for i in range(q_values.shape[-1]):
        heatmaps.append(cv2.applyColorMap(q_values[:, :, i], cv2.COLORMAP_JET))

    return heatmaps


def plot_vlaues(color: np.ndarray, depth: np.ndarray, values: np.ndarray, action) -> None:

    color = (color*255).astype(np.uint8)

    f = plt.figure(figsize=(10, 10))
    ax1 = f.add_subplot(3, 2, 1)
    ax1.imshow(color)
    ax2 = f.add_subplot(3, 2, 2)
    ax2.imshow(depth)

    heatmap = to_colorMap(values)

    for i, v in enumerate(heatmap):
        ax = f.add_subplot(3, 4, i+5)
        ax.imshow(v)
        ax = f.add_subplot(3, 4, i+9)
        merge = cv2.addWeighted(color, 1, v, 0.8, 0.0)
        if i == action[0]:
            print(action)
            merge = cv2.circle(
                merge, (action[1], action[2]), 3, (255, 255, 255), 2)
        ax.imshow(merge)
