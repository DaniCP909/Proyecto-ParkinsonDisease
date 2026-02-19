import torch
import numpy as np
from torch import randint
from torch.utils.data import Dataset
from utils.CustomMorphOps import fit_into_normalized_canvas, clean_and_refill, apply_saltpepper, shear, rotate
from utils.PatchesOps import patches_per_image, patch_generator
import cv2
from scipy.ndimage import grey_dilation

from domain.RepresentationType import RepresentationType
from domain.Patient import Patient
from utils.PatchesOps import patches_per_image, patch_generator

class PahawOfflineSimWindowDataset(Dataset):

    def __init__(
            self, 
            patients_dict: dict[int, Patient], 
            transform=None, 
            patch_w=150, 
            stepsize=75, 
            task_nums=[2],
            rep_type: RepresentationType=RepresentationType.SIMPLE_STROKE,
            target_mode = "binary",
            augment=False,
            ):
        self.patients = list(patients_dict.values())  
        self.rep_type = rep_type
        self.task_nums = task_nums
        self.target_mode = target_mode
        self.transform=transform
        self.patch_w = patch_w
        self.stepsize = stepsize
        self.kernel = np.ones((3, 3))
        self.augment=augment

        # Precomputamos todas las tareas
        self.samples = []
        for patient in self.patients:
            for t in task_nums:
                try:
                    task = patient.getTaskByTypeAndNum(rep_type, t)
                    self.samples.append((patient, task))
                except:
                    pass  # si falta la tarea (caso raro)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient, task = self.samples[idx]

        img = task.data
        original_w = img.shape[1]

        if self.augment:
            random_num = torch.randint(0, 4, (1,)).item()
            if random_num == 0:
                sh = np.random.uniform(-0.3, 0.3)
                img =shear(img, sh)
            if random_num == 1:
                angle = np.random.uniform(-3, 3)
                img = rotate(img, angle)
            if random_num == 2:
                img = grey_dilation(img, footprint=self.kernel)
        img = clean_and_refill(img, original_w)

        patches_list = patch_generator(
            img,
            n_patches=patches_per_image(original_w, self.patch_w, self.stepsize),
            patch_height=img.shape[0],
            patch_width=self.patch_w,
            stepsize=self.stepsize
        )
        patches_np = np.stack(patches_list, axis=0)
        patches = torch.from_numpy(patches_np).float()

        patches = patches.unsqueeze(1)

        # ---- TARGET
        if self.target_mode == "binary":
            y = torch.tensor(patient.pd_status, dtype=torch.long)
        elif self.target_mode == "severity":
            y = torch.tensor(min(patient.pd_years / 20.0, 1.0), dtype=torch.float32)
        elif self.target_mode == "multi_lable":
            y = torch.tensor(
                [float(patient.pd_status), patient.pd_years, min(patient.pd_years / 20.0, 1.0)],
                dtype=torch.float32
                )
        return patches, y, patient.id, idx, task.task_number
    

