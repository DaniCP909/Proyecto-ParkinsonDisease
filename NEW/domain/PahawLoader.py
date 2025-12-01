import os
import pandas
import cv2

from domain.Stroke import Stroke
from domain.Task import Task
from domain.RepresentationType import RepresentationType
from domain.Patient import Patient
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Y = 0
X = 1
TIMESTAMP = 2
BUTTON_STATE = 3
AZIMUTH = 4
ALTITUDE = 5
PRESSURE = 6

class PahawLoader:
    VALID_TASK_NUMS = tuple(range(1,9))
    """
    Loads PaHaW database organized with domain subclasses.
    Returns 2 dictionaries:
        - {id: (pd_status, pd_years)}
        - {id: Task}
    """

    def __init__(
            self,
    ):
        #self.subjects_pd_status_years_dict = {}
        #self.subject_tasks_dict = {}
        self.patients_dicts = {}

        self.global_max_w = 0
        self.global_max_h = 0
        self.global_max_w_task1 = 0
        self.global_max_h_task1 = 0

        #CACHE
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)

        #Pahaw data file
        pahaw_file_path = os.path.join(
            BASE_DIR,
            "PaHaW",              # o "PaHaW" según nombre exacto de tu carpeta
            "PaHaW_files",
            "corpus_PaHaW.xlsx"
        )
        pahaw_data_frame = pandas.read_excel(pahaw_file_path)

        #Pahaw tasks files
        task_file_path_start = os.path.join("PaHaW", "PaHaW_public")
        task_file_path_end = "_1.svc"

        #ids, status and years lists
        subjects_id_list = list(map(int, pahaw_data_frame["ID"].to_list()))
        subjects_pd_status_list = [
            0 if e == "H" else 1 for e in pahaw_data_frame["Disease"].to_list()
        ]
        subjects_pd_years_list = list(
            map(int, pahaw_data_frame["Length of PD"].fillna(0).to_list())
        )

        #Fill dicts
        #subjects_pd_status_years_dict = {}
        #subjects_tasks_dict = {}
        patients_dict = {}
        subject_i = 0
        while subject_i < len(subjects_id_list):
            subject_id = subjects_id_list[subject_i]
            pd_status_years = (
                subjects_pd_status_list[subject_i],
                subjects_pd_years_list[subject_i],
            )
            os.makedirs(os.path.join("generated_tasks", f"subject{subject_id}_GT{subjects_pd_status_list[subject_i]}"), exist_ok=True)

            new_patient = Patient(subject_id, pd_status_years[0], pd_status_years[1])

            for task_number in range(1,9):
                task_file_path_mid = os.path.join(
                    f"{subject_id:05d}", f"{subject_id:05d}__{task_number}"
                )
                task_file_path = os.path.join(
                    task_file_path_start, task_file_path_mid + task_file_path_end
                )
                #stroke management
                task_strokes_list = []
                all_coords = []
                if os.path.exists(task_file_path):
                    with open(task_file_path, encoding="utf-8") as task_file:
                        #skip first line
                        task_file.readline()
                        from_on_air = True

                        while True:
                            line = task_file.readline()
                            if not line:
                                break
                            coordinate = (
                                int(line.split()[X]),
                                int(line.split()[Y]),
                                int(line.split()[TIMESTAMP]),
                                int(line.split()[BUTTON_STATE]),
                                int(line.split()[AZIMUTH]),
                                int(line.split()[ALTITUDE]),
                                int(line.split()[PRESSURE]),
                            )
                            all_coords.append(coordinate)

                            #si coordenada en el aire
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

                #subjects_pd_status_years_dict[subject_id] = pd_status_years

                if task_strokes_list:    #si hay trazo

                    new_simple_task = Task(subject_id, task_number, task_strokes_list, all_coords, pd_status_years[0], rep_type=RepresentationType.SIMPLE_STROKE)
                    new_patient.addTask(new_simple_task)
                    #new_simple_task.generate_data()

                    if task_number == 1:
                        if new_simple_task.getWidth() > self.global_max_w_task1:
                            self.global_max_w_task1 = new_simple_task.getWidth()

                        if new_simple_task.getHeight() > self.global_max_h_task1:
                            self.global_max_h_task1 = new_simple_task.getHeight()
                    else:
                        if new_simple_task.getWidth() > self.global_max_w:
                            self.global_max_w = new_simple_task.getWidth()

                        if new_simple_task.getHeight() > self.global_max_h:
                            self.global_max_h = new_simple_task.getHeight()

                    new_enhanced_task = Task(subject_id, task_number, task_strokes_list, all_coords, pd_status_years[0], rep_type=RepresentationType.ENHANCED_STROKE)
                    new_patient.addTask(new_enhanced_task)
                    #new_enhanced_task.generate_data()

                    new_multichannel_task = Task(subject_id, task_number, task_strokes_list, all_coords, pd_status_years[0], rep_type=RepresentationType.MULTICHANNEL)
                    new_patient.addTask(new_multichannel_task)
                    #new_multichannel_task.generate_data()

                    new_online_signal_task = Task(subject_id, task_number, task_strokes_list, all_coords, pd_status_years[0], rep_type=RepresentationType.ONLINE_SIGNAL)
                    new_patient.addTask(new_online_signal_task)

                    patients_dict[subject_id] = new_patient

                    #debug = (new_enhanced_task.data * 255).astype(np.uint8)

                    #cv2.imwrite(f"tareas_generadas/img_{subject_id}_{task_number}.png", debug)

                        #print(f"[CACHE] Guardada tarea {task_number} del sujeto {subject_id}")
                    #if subject_id not in subjects_tasks_dict:
                    #    subjects_tasks_dict[subject_id] = {}
                    #subjects_tasks_dict[subject_id][task_number] = new_simple_task
                else:
                    print(f"Tarea vacía para Sujeto {subject_id}, Tarea {task_number}, se omite.")
            subject_i += 1
        #print(list(zip(subjects_id_list, subjects_pd_status_list)))
        #print(subjects_pd_status_years_dict)

        print(f"MEDIDAS FINALES: {self.global_max_w}, {self.global_max_h}")

        #subject_ii = 0
        #while subject_ii < len(subjects_id_list):
        #    subject_id = subjects_id_list[subject_ii]
        #    for task_number in range(2,9):
        #        task = subjects_tasks_dict[subject_id].get(task_number)
        #        if task is not None:
        #            task.generate_data(self.global_max_h, self.global_max_w)
        #    subject_ii += 1
        for patient_id, patient in patients_dict.items():
            tasks_dicts_dict = patient.getTasksListsDict()
            for rep_key in tasks_dicts_dict.keys():
                for task_key in tasks_dicts_dict[rep_key].keys():
                    task = tasks_dicts_dict[rep_key][task_key]
                    if task_key == 1:
                        task.generate_data(self.global_max_h_task1, self.global_max_w_task1, task1=True)
                    else:
                        task.generate_data(self.global_max_h, self.global_max_w)
                #print(f"Paciente {patient_id} len {key}: {len(tasks_lists)}")

        #self.subjects_pd_status_years_dict = subjects_pd_status_years_dict
        #self.subjects_tasks_dict = subjects_tasks_dict
        self.patients_dicts = patients_dict


    def load(self):
        return self.patients_dicts
    
    def loadCustomSubset(self, rep_type: RepresentationType, task_nums: list[int]):
        invalid = set(task_nums) - set(self.VALID_TASK_NUMS)
        if invalid:
            raise ValueError(f"Invalid task {invalid}")
        
        task_nums = sorted(task_nums)
        
        subset = []
        for t in task_nums:
            for patient in self.patients_dicts.values():
                subset.append(patient.getTaskByTypeAndNum(rep_type, t))
        return subset