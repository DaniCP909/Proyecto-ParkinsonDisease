from torch.utils.data import Dataset
import torch

from domain.Patient import Patient
from domain.RepresentationType import RepresentationType

class PahawOfflineSimDataset(Dataset):

    def __init__(
            self, 
            patients_dict: dict[int, Patient], 
            transform=None, 
            patch_w=200, 
            stepsize=2, 
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

        # X
        x = torch.tensor(task.data, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (1, H, W)

        # Y
        if self.target_mode == "binary":
            y = torch.tensor(patient.pd_status, dtype=torch.long)

        elif self.target_mode == "severity":
            y = torch.tensor(min(patient.pd_years / 20.0, 1.0), dtype=torch.float32)

        elif self.target_mode == "multi_label":
            vec = (
                float(patient.pd_status),
                min(patient.pd_years / 20.0, 1.0)
            )
            y = torch.tensor(vec, dtype=torch.float32)

        # Devolver 4 valores como espera tu pipeline
        return x, y, patient.id, idx
    

