import torch
import numpy as np
from torch.utils.data import Dataset
from utils.CustomMorphOps import fit_into_normalized_canvas
from utils.PatchesOps import patches_per_image, patch_generator
import cv2


class PahawOfflineSimDataset(Dataset):

    def __init__(self, data: list[tuple], device, transform=None, patch_w=200, stepsize=2, id_list=[]):
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
        self.id_list = id_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.tasks[idx].getCanvases()['stroke']
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        fit_image = fit_into_normalized_canvas(image, self.max_h, self.max_w)
        png_img = (fit_image * 255).astype(np.uint8)
        cv2.imwrite(f"results/indexed_tagged_tasks/2/fit{self.id_list[idx]}_{label}.png", png_img)

        n_patches = patches_per_image(fit_image.shape[1], patch_width=self.patch_w, stepsize=self.setpsize)
        patches = patch_generator(fit_image, device=self.device, n_patches=n_patches,
                                  patch_height=fit_image.shape[0],
                                  patch_width=self.patch_w,
                                  stepsize=self.setpsize)
        patches_tensor = torch.tensor(np.stack(patches), dtype=torch.float32).unsqueeze(1)
        return patches_tensor, label

