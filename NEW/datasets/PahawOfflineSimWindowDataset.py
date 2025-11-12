import torch
import numpy as np
from torch.utils.data import Dataset
from utils.CustomMorphOps import fit_into_normalized_canvas, clean_and_refill
from utils.PatchesOps import patches_per_image, patch_generator
import cv2


class PahawOfflineSimWindowDataset(Dataset):

    def __init__(self, data: list[tuple], device, transform=None, patch_w=200, stepsize=2, task_num=2):
        zip_tasks, zip_labels = zip(*data)
        self.tasks = list(zip_tasks)
        self.labels = list(zip_labels)
        self.transform = transform
        self.device = device
        self.patch_w = patch_w
        self.setpsize = stepsize
        self.max_h = float('-inf')
        self.max_w = float('-inf')
        self.task_num=task_num
        for task in self.tasks:
            self.max_h = max(self.max_h, task.getHeight())
            self.max_w = max(self.max_w, task.getWidth())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.tasks[idx].getCanvases()['stroke']
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        fit_image = fit_into_normalized_canvas(image, self.max_h, self.max_w)

        clean_img = clean_and_refill(fit_image)

        n_patches = patches_per_image(clean_img.shape[1], patch_width=self.patch_w, stepsize=self.setpsize)
        patches = patch_generator(clean_img, device=self.device, n_patches=n_patches,
                                  patch_height=clean_img.shape[0],
                                  patch_width=self.patch_w,
                                  stepsize=self.setpsize)
        patches_tensor = torch.tensor(np.stack(patches), dtype=torch.float32).unsqueeze(1)
        return patches_tensor, label
    

