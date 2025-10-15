# subset_utils.py
import random

def build_subsets(subjects_pd_status_years, subjects_tasks, args=None, min_task=5, max_task=6):
    subjects_ids = list(subjects_tasks.keys())

    h_id_list, pd_id_list = [], []

    for subject_id in subjects_ids:
        if subjects_pd_status_years[subject_id][0] == 0:
            h_id_list.append(subject_id)
        else:
            pd_id_list.append(subject_id)
    
    random.Random(args.seed).shuffle(h_id_list)
    random.Random(args.seed).shuffle(pd_id_list)
    
    raw_mix = []

    for i, val in enumerate(h_id_list):
        raw_mix.append(val)
        if i < len(pd_id_list):
            raw_mix.append(pd_id_list[i])
    if len(pd_id_list) > len(h_id_list):
        raw_mix.extend(pd_id_list[len(h_id_list)])

    validate_id_list = raw_mix[55:]
    train_id_list = raw_mix[:55]

    repeticiones = [val for val in validate_id_list if val in train_id_list]
    print(f"REPS: {repeticiones}")

    print("ID finales train")
    print(train_id_list)
    print("ID finales validate")
    print(validate_id_list)

    #Create data sets (IMG, LABEL) structure
    train_label_img, validate_label_img = [], []
    #Train Validate and overfit
    for subject_id in train_id_list:
        for task_num in range(min_task, max_task):
            task = subjects_tasks[subject_id].get(task_num)
            if task is not None:
                train_label_img.append((task, subjects_pd_status_years[subject_id][0]))
    for subject_id in validate_id_list:
        for task_num in range(min_task, max_task):
            task = subjects_tasks[subject_id].get(task_num)
            if task is not None:
                validate_label_img.append((task, subjects_pd_status_years[subject_id][0]))
    print("train_label_img")
    print(train_label_img)
    print("train_label_img")
    print(validate_label_img)
    
    return train_id_list, train_label_img, validate_id_list, validate_label_img

def build_overfit_subsets(subjects_pd_status_years, subjects_tasks, args=None, task_num=2):
    subjects_ids = list(subjects_tasks.keys())

    h_id_list, pd_id_list, overfit_id_list = [], [], []

    for subject_id in subjects_ids:
        if subjects_pd_status_years[subject_id][0] == 0:
            h_id_list.append(subject_id)
        else:
            pd_id_list.append(subject_id)
    
    random.Random(args.seed).shuffle(h_id_list)
    random.Random(args.seed).shuffle(pd_id_list)

    #[ID LIST]Init overfit ids lists
    for i, val in enumerate(h_id_list):
        overfit_id_list.append(val)
        if i < len(pd_id_list):
            overfit_id_list.append(pd_id_list[i])
    if len(pd_id_list) > len(h_id_list):
        overfit_id_list.extend(pd_id_list[len(h_id_list):])

    overfit_train_ids = overfit_id_list
    overfit_validate_ids = overfit_id_list

    repeticiones = [val for val in overfit_validate_ids if val in overfit_train_ids]
    print(f"REPS: {repeticiones}")

    #Create data sets (IMG, LABEL) structure
    overfit_train_label_img, overfit_validate_label_img = [], []
    #Train Validate and overfit
    for subject_id in overfit_id_list:
        task = subjects_tasks[subject_id].get(task_num)
        if task is not None:
            label = subjects_pd_status_years[subject_id][0]
            overfit_train_label_img.append((task, label))
            overfit_validate_label_img.append((task, label))
    print("overfit_train_label_img")
    print(overfit_train_label_img[:2])
    print("overfit_validate_label_img")
    print(overfit_validate_label_img[:2])
                    
    return overfit_train_ids, overfit_train_label_img, overfit_validate_ids, overfit_validate_label_img

