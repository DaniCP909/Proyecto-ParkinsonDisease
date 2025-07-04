import torch
import numpy as np
from torch.utils.data import Dataset
from CustomMorphOps import fit_into_normalized_canvas

class PahawOfflineSimDataset(Dataset):

    def __init__(self, data: list[tuple], device, transform=None, patch_w=200, stepsize=2):
        zip_tasks, zip_labels = zip(*data)
        self.tasks = list(zip_tasks)
        self.labels = list(zip_labels)
        self.transform = transform
        self.device = device
        self.patch_w = patch_w
        self.setpsize = stepsize
        self.max_h = float('-inf')
        self.max_w = float('-inf')
        for task in self.tasks:
            self.max_h = max(self.max_h, task.getHeight())
            self.max_w = max(self.max_w, task.getWidth())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.tasks[idx].getImage()
        labels = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        fit_image = fit_into_normalized_canvas(image, self.max_h, self.max_w)

        n_patches = patches_per_image(fit_image.shape[1], patch_width=self.patch_w, stepsize=self.setpsize)
        patches = patch_generator(fit_image, device=self.device, n_patches=n_patches,
                                  patch_height=fit_image.shape[0],
                                  patch_width=self.patch_w,
                                  stepsize=self.setpsize)
        patches_tensor = torch.tensor(np.stack(patches), dtype=torch.float32).unsqueeze(1)
        return patches_tensor, labels


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
