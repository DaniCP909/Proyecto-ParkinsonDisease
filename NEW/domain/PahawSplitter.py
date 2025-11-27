import random
from domain.Patient import Patient

class PahawSplitter:
    def __init__(self, patients_dicts: dict[int, Patient]):
        self.patients = patients_dicts

    def stratified_split(self, val_ratio=0.2, seed=42):
        random.seed(seed)

        healthy = [p for p in self.patients.values() if p.pd_status == 0]
        parkinson = [p for p in self.patients.values() if p.pd_status == 1]

        random.shuffle(healthy)
        random.shuffle(parkinson)

        n_val_h = int(len(healthy) * val_ratio)
        n_val_p = int(len(parkinson) * val_ratio)

        val = healthy[:n_val_h] + parkinson[:n_val_p]
        train = healthy[n_val_h:] + parkinson[n_val_p:]

        random.shuffle(val)
        random.shuffle(train)

        train_dict = {p.id: p for p in train}
        val_dict = {p.id: p for p in val}

        return train_dict, val_dict