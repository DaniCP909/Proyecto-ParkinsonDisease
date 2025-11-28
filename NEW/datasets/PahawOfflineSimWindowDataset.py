import torch
import numpy as np
from torch.utils.data import Dataset
from utils.CustomMorphOps import fit_into_normalized_canvas, clean_and_refill
from utils.PatchesOps import patches_per_image, patch_generator
import cv2

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
            target_mode = "binary"
            ):
        self.patients = list(patients_dict.values())  
        self.rep_type = rep_type
        self.task_nums = task_nums
        self.target_mode = target_mode
        self.transform=transform
        self.patch_w = patch_w
        self.stepsize = stepsize

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

        patches_list = patch_generator(
            img,
            n_patches=patches_per_image(img.shape[1], self.patch_w, self.stepsize),
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
    

