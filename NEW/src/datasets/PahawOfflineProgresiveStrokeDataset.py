from torch.utils.data import Dataset
import torch
from domain.Patient import Patient
from domain.RepresentationType import RepresentationType
from utils.CustomMorphOps import bresenham_line, normalize, fit_into_normalized_canvas
import cv2

import numpy as np

def _rep_enhanced_stroke(self, min_thickness = 2, max_thickness = 10, min_dark_factor = 0.7, max_dark_factor = 0.99):
        final_w = int(self.max_vals['x_surface'] - self.min_vals['x_surface'])
        final_h = int(self.max_vals['y_surface'] - self.min_vals['y_surface'])
        if final_h == 0 or final_w == 0:
            print(f"[DEBUG] Subject {self.subject_id} task {self.task_number} -> H: {final_h}, W: {final_w}")
        canvas = np.ones((final_h, final_w), dtype=np.float32)
        for letters_set in self.letters_sets_list:
            for stroke in letters_set.strokes_list:
                stroke_x_list = stroke.get_x_coordinates_list()
                stroke_y_list = stroke.get_y_coordinates_list()
                normalized_x = [x - self.min_vals['x_surface'] for x in stroke_x_list]
                normalized_y = [y - self.min_vals['y_surface'] for y in stroke_y_list]
                altitudes = stroke.getAltitudes()
                normalized_altitudes = normalize(altitudes)
                pressures = stroke.getPressures()
                normalized_pressures = normalize(pressures)
                for i in range(len(stroke_x_list) -1):
                    darkening_factor = min_dark_factor + (max_dark_factor - min_dark_factor) * (1 - normalized_pressures[i])
                    thickness_factor = min_thickness + (max_thickness - min_thickness) * (1 - normalized_altitudes[i])
                    pixels = bresenham_line(
                        normalized_x[i],
                        normalized_y[i],
                        normalized_x[i+1],
                        normalized_y[i+1],
                        height=final_h,
                        width=final_w,
                        thickness=int(thickness_factor),
                        )
                    for y, x in pixels:
                        canvas[y, x] *= darkening_factor

class PahawOfflineProgresiveStrokeDataset(Dataset):
    
    def __init__(
            self, 
            patients_dict: dict[int, Patient],
            global_h,
            global_w,
            transform=None, 
            n_stages = 10,
            task_nums=[2],
            rep_type: RepresentationType=RepresentationType.ONLINE_SIGNAL,
            augment=False,
            ):
        self.patients = list(patients_dict.values())  
        self.rep_type = rep_type
        self.task_nums = task_nums
        self.transform=transform
        self.n_stages = n_stages
        self.kernel = np.ones((3, 3))
        self.augment=augment
        self.global_h = global_h
        self.global_w = global_w

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

        min_thickness = 2
        max_thickness = 10
        min_dark_factor = 0.7
        max_dark_factor = 0.99

        data = task.letters_sets_list

        min_vals = task.get_min_vals()
        max_vals = task.get_max_vals()

        final_w = int(max_vals['x_surface'] - min_vals['x_surface'])
        final_h = int(max_vals['y_surface'] - min_vals['y_surface'])

        all_coordinates = []

        stages_images = []

        for letters_set in data:
            for stroke in letters_set.strokes_list:
                stroke_x_list = stroke.get_x_coordinates_list()
                stroke_y_list = stroke.get_y_coordinates_list()
                normalized_x = [x - min_vals['x_surface'] for x in stroke_x_list]
                normalized_y = [y - min_vals['y_surface'] for y in stroke_y_list]
                altitudes = stroke.getAltitudes()
                normalized_altitudes = normalize(altitudes)
                pressures = stroke.getPressures()
                normalized_pressures = normalize(pressures)
                for i in range(len(stroke_x_list) -1):
                    darkening_factor = min_dark_factor + (max_dark_factor - min_dark_factor) * (1 - normalized_pressures[i])
                    thickness_factor = min_thickness + (max_thickness - min_thickness) * (1 - normalized_altitudes[i])
                    super_coord = (
                        normalized_x[i],
                        normalized_y[i],
                        normalized_x[i+1],
                        normalized_y[i+1],
                        final_h,
                        final_w,
                        int(thickness_factor),
                        darkening_factor,
                    )
                    all_coordinates.append(super_coord)

        n_coords = len(all_coordinates) // self.n_stages

        for i in range(self.n_stages):
            canvas = np.ones((final_h, final_w), dtype=np.float32)
            end_idx = min((i + 1) * n_coords, len(all_coordinates))
            for j in all_coordinates[:end_idx]:
                pixels_to_draw = bresenham_line(*j[:7])
                darkening_factor = j[7]
                for y, x in pixels_to_draw:
                            canvas[y, x] *= darkening_factor
            flip_img = cv2.flip(canvas, 0)
            negative_img = 1.0 - flip_img
            resized = fit_into_normalized_canvas(negative_img, self.global_h, self.global_w)
            stages_images.append(resized)
        
        stages_array = np.stack(stages_images, axis=0)   # (T, H, W)
        stages_tensor = torch.from_numpy(stages_array).float().unsqueeze(1)  # (T, 1, H, W)

        y = torch.tensor(patient.pd_status, dtype=torch.long)

        return stages_tensor, y, patient.id, idx, task.task_number
