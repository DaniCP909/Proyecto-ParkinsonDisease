import os
import pandas
import cv2

from domain.Stroke import Stroke
from domain.Task import Task
from domain.TaskEmothaw import TaskEmothaw
from domain.RepresentationType import RepresentationType
from domain.EmothawPatient import EmothawPatient
import numpy as np

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

Y = 0
X = 1
TIMESTAMP = 2
BUTTON_STATE = 3
AZIMUTH = 4
ALTITUDE = 5
PRESSURE = 6

class EmothawLoader:
    VALID_TASK_NUMS = tuple(range(1,9))
    """
    Loads EMOTHaw database organized with domain subclasses.
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

        #Pahaw data file
        self.emothaw_root_path = os.path.join(
            BASE_DIR,
            "EMOTHaw",

        )
        self.emothaw_info_file_path = os.path.join(
            self.emothaw_root_path,
            "DASS_scores.xls"
        )

        emothaw_df = pandas.read_excel(self.emothaw_info_file_path)

        subjects_id_list = list(range(len(emothaw_df)))
        print(subjects_id_list)

        tasks = (3, 7)

        collections = (
            ("Collection1", range(1, 46)),
            ("Collection2", range(1, 85)),
        )

        self.id_equivalent = [(collection_name,subject_id) for collection_name,subjects in collections for subject_id in subjects]

        patients_dict = {}
        
        for subject_id, (collection_name, user_number) in enumerate(self.id_equivalent):
            row = emothaw_df.iloc[subject_id]
            d, a, s = row[["depression", "anxiety", "stress"]]

            new_patient = EmothawPatient(subject_id, d, a, s)

            for task in tasks:
                file_path = os.path.join(
                    self.emothaw_root_path,
                    collection_name,
                    f"user{user_number:05d}",
                    "session00001",
                    f"u{user_number:05d}s00001_hw{task:05d}.svc"
                )
                task_strokes_list = []
                all_coords = []
                if os.path.exists(file_path):
                    with open(file_path, encoding="utf-8") as task_file:
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
                    print(f"Archivo no encontrado: {file_path}, se omite.")

                if task_strokes_list:

                    new_simple_task = TaskEmothaw(subject_id, task, task_strokes_list, all_coords, d, a, s, RepresentationType.SIMPLE_STROKE, cache_base_dir="emothaw_images")
                    new_patient.addTask(new_simple_task)

                    patients_dict[subject_id] = new_patient
            self.patients_dict = patients_dict

    def load(self):
        return self.patients_dict



#        for collection_name, subjects in collections:
#            for subject_id in subjects:
#                for task in tasks:
#
#                    aux_path = os.path.join(
#                        self.emothaw_root_path,
#                        collection_name,
#                        f"user{subject_id:05d}",
#                        "session00001",
#                        f"u{subject_id:05d}s00001_hw{task:05d}.svc"
#                    )
#
#                    if not os.path.exists(aux_path):
#                        print(f"No existe: {aux_path}")
#                        continue
#
#                    all_files_path.append(aux_path)
#        
#        for file_path in all_files_path:
#            task_strokes_list = []
#            all_coords = []
#            if os.path.exists(file_path):
#                with open(file_path, encoding="utf-8") as file:
#
#                    #skip first line
#                    file.readline()
#                    from_on_air = True
#
#                    while True:
#                        line = file.readline()
#                        if not line:
#                            break
#
#                        coordinate = (
#                            int(line.split()[X]),
#                            int(line.split()[Y]),
#                            int(line.split()[TIMESTAMP]),
#                            int(line.split()[BUTTON_STATE]),
#                            int(line.split()[AZIMUTH]),
#                            int(line.split()[ALTITUDE]),
#                            int(line.split()[PRESSURE]),
#                        )
#                        all_coords.append(coordinate)
#
#                        #si coordenada en el aire
#                        if line.split()[3] == "1":
#                            if from_on_air:
#                                task_strokes_list.append(Stroke(coordinate))
#                                from_on_air = False
#                            else:
#                                task_strokes_list[-1].append(coordinate)
#                        else:
#                            from_on_air = True
                
#        pahaw_data_frame = pandas.read_excel(pahaw_file_path)
#
#        #Pahaw tasks files
#        task_file_path_start = os.path.join(BASE_DIR, "PaHaW", "PaHaW_public")
#        task_file_path_end = "_1.svc"
#
#        #ids, status and years lists
#        subjects_id_list = list(map(int, pahaw_data_frame["ID"].to_list()))
#        subjects_pd_status_list = [
#            0 if e == "H" else 1 for e in pahaw_data_frame["Disease"].to_list()
#        ]
#        subjects_pd_years_list = list(
#            map(int, pahaw_data_frame["Length of PD"].fillna(0).to_list())
#        )
#
#        patients_dict = {}
#        subject_i = 0
#        while subject_i < len(subjects_id_list):
#            subject_id = subjects_id_list[subject_i]
#            pd_status_years = (
#                subjects_pd_status_list[subject_i],
#                subjects_pd_years_list[subject_i],
#            )
#
#            new_patient = Patient(subject_id, pd_status_years[0], pd_status_years[1])
#
#            for task_number in range(1,9):
#                task_file_path_mid = os.path.join(
#                    f"{subject_id:05d}", f"{subject_id:05d}__{task_number}"
#                )
#                task_file_path = os.path.join(
#                    task_file_path_start, task_file_path_mid + task_file_path_end
#                )
#                #stroke management
#                task_strokes_list = []
#                all_coords = []
#                if os.path.exists(task_file_path):
#                    with open(task_file_path, encoding="utf-8") as task_file:
#                        #skip first line
#                        task_file.readline()
#                        from_on_air = True
#
#                        while True:
#                            line = task_file.readline()
#                            if not line:
#                                break
#                            coordinate = (
#                                int(line.split()[X]),
#                                int(line.split()[Y]),
#                                int(line.split()[TIMESTAMP]),
#                                int(line.split()[BUTTON_STATE]),
#                                int(line.split()[AZIMUTH]),
#                                int(line.split()[ALTITUDE]),
#                                int(line.split()[PRESSURE]),
#                            )
#                            all_coords.append(coordinate)
#
#                            #si coordenada en el aire
#                            if line.split()[3] == "1":
#                                if from_on_air:
#                                    task_strokes_list.append(Stroke(coordinate))
#                                    from_on_air = False
#                                else:
#                                    task_strokes_list[-1].append(coordinate)
#                            else:
#                                from_on_air = True
#                else:
#                    print(f"Archivo no encontrado: {task_file_path}, se omite.")
#
#
#                if task_strokes_list:    #si hay trazo
#
#                    new_simple_task = Task(subject_id, task_number, task_strokes_list, all_coords, pd_status_years[0], rep_type=RepresentationType.SIMPLE_STROKE)
#                    new_patient.addTask(new_simple_task)
#
#                    if task_number == 1:
#                        if new_simple_task.getWidth() > self.global_max_w_task1:
#                            self.global_max_w_task1 = new_simple_task.getWidth()
#
#                        if new_simple_task.getHeight() > self.global_max_h_task1:
#                            self.global_max_h_task1 = new_simple_task.getHeight()
#                    else:
#                        if new_simple_task.getHeight() > self.global_max_h:
#                            self.global_max_h = new_simple_task.getHeight()
#                            self.global_max_h_id = subject_id
#
#                    new_enhanced_task = Task(subject_id, task_number, task_strokes_list, all_coords, pd_status_years[0], rep_type=RepresentationType.ENHANCED_STROKE)
#                    new_patient.addTask(new_enhanced_task)
#
#                    new_multichannel_task = Task(subject_id, task_number, task_strokes_list, all_coords, pd_status_years[0], rep_type=RepresentationType.MULTICHANNEL)
#                    new_patient.addTask(new_multichannel_task)
#
#                    new_online_signal_task = Task(subject_id, task_number, task_strokes_list, all_coords, pd_status_years[0], rep_type=RepresentationType.ONLINE_SIGNAL)
#                    new_patient.addTask(new_online_signal_task)
#
#                    patients_dict[subject_id] = new_patient
#
#                else:
#                    print(f"Tarea vacía para Sujeto {subject_id}, Tarea {task_number}, se omite.")
#            subject_i += 1
#
#
#        for patient_id, patient in patients_dict.items():
#            tasks_dicts_dict = patient.getTasksListsDict()
#            rep_key =RepresentationType.SIMPLE_STROKE
#            for task_key in tasks_dicts_dict[rep_key].keys():
#                if task_key != 1:
#                    task = tasks_dicts_dict[rep_key][task_key]
#                    tw = task.getWidth()
#                    th = task.getHeight()
#                    #print(f"ID: {patient_id}|task{task_key}: original_h: {th}, original_w: {tw}")
#                    h_resized_factor = self.global_max_h / th
#                    final_w = int(np.ceil(tw * h_resized_factor))
#                    if final_w > self.global_max_resized_w:
#                        self.global_max_resized_w = final_w
#                        self.global_max_resized_w_id = patient_id
#                        self.w_task = task_key
#
#        self.patients_dicts = patients_dict
#
#        for patient_id, patient in patients_dict.items():
#            tasks_dicts_dict = patient.getTasksListsDict()
#            for rep_key in tasks_dicts_dict.keys():
#                for task_key in tasks_dicts_dict[rep_key].keys():
#                    task = tasks_dicts_dict[rep_key][task_key]
#                    if task_key == 1:
#                        task.generate_data(self.global_max_h_task1, self.global_max_w_task1, task1=True)
#                    else:
#                        task.generate_data(self.global_max_h, self.global_max_resized_w)
#        self.patients_dicts = patients_dict
#
#        print(f"MEDIDAS FINALES: H: {self.global_max_h} patient: {self.global_max_h_id}, W: {self.global_max_resized_w} patient: {self.global_max_resized_w_id} | task: {self.w_task}")
#
#
#    def load(self):
#        return self.patients_dicts
#    
#    def loadCustomSubset(self, rep_type: RepresentationType, task_nums: list[int]):
#        invalid = set(task_nums) - set(self.VALID_TASK_NUMS)
#        if invalid:
#            raise ValueError(f"Invalid task {invalid}")
#        
#        task_nums = sorted(task_nums)
#        
#        subset = []
#        for t in task_nums:
#            for patient in self.patients_dicts.values():
#                subset.append(patient.getTaskByTypeAndNum(rep_type, t))
#        return subset