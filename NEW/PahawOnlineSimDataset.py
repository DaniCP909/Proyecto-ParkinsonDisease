import torch
import numpy as np
from torch.utils.data import Dataset
from CustomMorphOps import fit_into_normalized_canvas

class PahawOnlineSimDataset(Dataset):

    def __init__(self, data: list[tuple], device):
        zip_tasks, zip_labels = zip(*data)
        self.tasks = list(zip_tasks)
        self.labels = list(zip_labels)
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        all_coords = self.tasks[idx].getAllCordinates()
        labels = self.labels[idx]

        coords_tensor = torch.tensor(all_coords, dtype=torch.float32)
        return coords_tensor, labels


def patches_per_image(image_width, patch_width=10, stepsize=2):
    return int((image_width - patch_width)/stepsize + 1)

def patch_generator(image, device, n_patches=1, patch_height=48, patch_width=10, stepsize=2):
    
    H, W = image.shape
    patches = []
    for p in range(n_patches):
        start_x = p * stepsize
        patch = image[:, start_x: start_x + patch_width]
        patches.append(patch)
    return patches
