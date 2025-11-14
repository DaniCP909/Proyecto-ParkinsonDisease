import math
import matplotlib
import os
import pandas
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle

from utils.CustomMorphOps import bresenham_line, normalize
from domain.Task import Task


def load_data() -> tuple[dict[int, tuple[int, int]], dict[int, dict[int, Task]]]:
    """Devuelve dos diccionarios, ambos con la ID del sujeto como clave. En el primero
    el valor es una tupla con el estado PD y los años PD y en el otro una lista con las
    tareas 2, 3 y 4 del sujeto.
    """
    
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)

    pahaw_file_path = os.path.join("PaHaW", "PaHaW_files", "corpus_PaHaW.xlsx")
    pahaw_data_frame = pandas.read_excel(pahaw_file_path)
    task_file_path_start = os.path.join("PaHaW", "PaHaW_public")
    task_file_path_end = "_1.svc"
    subjects_id_list = list(map(int, pahaw_data_frame["ID"].to_list()))
    subjects_pd_status_list = [
        0 if e == "H" else 1 for e in pahaw_data_frame["Disease"].to_list()
    ]
    subject_pd_years_list = list(
        map(int, pahaw_data_frame["Length of PD"].fillna(0).to_list())
    )

    subjects_pd_status_years_dict = {}
    subjects_tasks_dict = {}
    subject_i = 0
    while subject_i < len(subjects_id_list):
        subject_id = subjects_id_list[subject_i]
        pd_status_years = (
            subjects_pd_status_list[subject_i],
            subject_pd_years_list[subject_i],
        )
        os.makedirs(os.path.join("tareas_generadas", f"sujeto{subject_id}_GT{subjects_pd_status_list[subject_i]}"), exist_ok=True)
        for task_number in range(1, 9):
            task_file_path_mid = os.path.join(
                f"{subject_id:05d}", f"{subject_id:05d}__{task_number}"
            )
            task_file_path = os.path.join(
                task_file_path_start, task_file_path_mid + task_file_path_end
            )
            task_strokes_list = []
            all_coords = []
            if os.path.exists(task_file_path):
                with open(task_file_path, encoding="utf-8") as task_file:
                    # Se salta la primera línea.
                    task_file.readline()
                    from_on_air = True

                    while True:
                        line = task_file.readline()
                        if not line:
                            break
                        coordinate = (
                            int(line.split()[1]),
                            int(line.split()[0]),
                            int(line.split()[2]),
                            int(line.split()[3]),
                            int(line.split()[4]),
                            int(line.split()[5]),
                            int(line.split()[6]),
                        )
                        all_coords.append(coordinate)
                        # Si la coordenada está sobre el papel.
                        if line.split()[3] == "1":
                            if from_on_air:
                                task_strokes_list.append(Stroke(coordinate))
                                from_on_air = False
                            else:
                                task_strokes_list[-1].append(coordinate)
                        else:
                            from_on_air = True
            else:
                print(f"Archivo no encontrado: {task_file_path}, se omite.")

            subjects_pd_status_years_dict[subject_id] = pd_status_years

            if task_strokes_list:  # Solo si hay trazos
                cache_path = os.path.join(cache_dir, f"{subject_id}_task{task_number}.pkl")

                if os.path.exists(cache_path):
                    with open(cache_path, "rb") as f:
                        new_task = pickle.load(f)
                    #print(f"[CACHE] Cargada tarea {task_number} del sujeto {subject_id}")
                else:
                    new_task = Task(subject_id, task_number, task_strokes_list, all_coords, pd_status_years[0])

                    new_task.plot_task(subdir=f"sujeto{subject_id}_GT{subjects_pd_status_list[subject_i]}")

                    with open(cache_path, "wb") as f:
                        pickle.dump(new_task, f, protocol=pickle.HIGHEST_PROTOCOL)
                    #print(f"[CACHE] Guardada tarea {task_number} del sujeto {subject_id}")
                if subject_id not in subjects_tasks_dict:
                    subjects_tasks_dict[subject_id] = {}
                subjects_tasks_dict[subject_id][task_number] = new_task
            else:
                print(f"Tarea vacía para Sujeto {subject_id}, Tarea {task_number}, se omite.")
        subject_i += 1
    #print(list(zip(subjects_id_list, subjects_pd_status_list)))
    #print(subjects_pd_status_years_dict)

    return subjects_pd_status_years_dict, subjects_tasks_dict

 