import os
import pandas
import pickle

from domain.Stroke import Stroke
from domain.Task import Task

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Y = 0
X = 1
TIMESTAMP = 2
BUTTON_STATE = 3
AZIMUTH = 4
ALTITUDE = 5
PRESSURE = 6

class PahawLoader:
    """
    Loads PaHaW database organized with domain subclasses.
    Returns 2 dictionaries:
        - {id: (pd_status, pd_years)}
        - {id: Task}
    """

    def __init__(
            self,
    ):
        self.subjects_pd_status_years_dict = {}
        self.subject_tasks_dict = {}

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
        subjects_pd_status_years_dict = {}
        subjects_tasks_dict = {}
        subject_i = 0
        while subject_i < len(subjects_id_list):
            subject_id = subjects_id_list[subject_i]
            pd_status_years = (
                subjects_pd_status_list[subject_i],
                subjects_pd_years_list[subject_i],
            )
            os.makedirs(os.path.join("generated_tasks", f"subject{subject_id}_GT{subjects_pd_status_list[subject_i]}"), exist_ok=True)

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

                subjects_pd_status_years_dict[subject_id] = pd_status_years

                if task_strokes_list:    #si hay trazo
                    cache_path = os.path.join(cache_dir, f"{subject_id}_task{task_number}.pkl")

                    if os.path.exists(cache_path):
                        with open(cache_path, "rb") as f:
                            new_task = pickle.load(f)
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

        self.subjects_pd_status_years_dict = subjects_pd_status_years_dict
        self.subjects_tasks_dict = subjects_tasks_dict


    def load(self):
        return self.subjects_pd_status_years_dict, self.subjects_tasks_dict