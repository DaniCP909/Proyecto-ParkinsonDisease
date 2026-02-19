# pipeline.py
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, Subset
import models.OfflineCnnLstm as cnnlstm
import models.OfflineCnnOnly as cnnonly
from datasets.PahawOfflineSimDataset import PahawOfflineSimDataset
from datasets.PahawOfflineSimWindowDataset import PahawOfflineSimWindowDataset
import os
import cv2
import numpy as np
import csv
from datetime import datetime

from domain.RepresentationType import RepresentationType
from domain.Patient import Patient

from utils.EarlyStopping import EarlyStopping

wrong_predicts = {
    "2": [5, 16, 17, 21, 24, 27, 31, 32, 35, 36, 38, 39, 46, 52, 56, 60, 64, 65, 70, 71, 72], #batch_size=2
    "3": [4, 5, 9, 19, 21, 27, 31, 32, 35, 36, 39, 42, 44, 47, 50, 56, 65, 71],
    "4": [1, 5, 7, 11, 12, 13, 17, 19, 23, 27, 34, 35, 40, 48, 55, 61, 62, 67],
    "5": [1, 2, 3, 4, 7, 8, 10, 15, 16, 18, 20, 23, 24, 25, 28, 29, 31, 33, 34, 37, 49, 50, 53, 56, 59, 60, 62, 63, 64, 66, 67, 69, 70, 72, 73],
    "6": [0, 2, 3, 5, 9, 10, 11, 13, 14, 16, 17, 19, 21, 22, 27, 29, 30, 32, 34, 36, 38, 40, 41, 42, 44, 45, 46, 47, 52, 54, 56, 65, 66, 68, 71, 72, 74],
    "7": [1, 2, 7, 8, 12, 14, 16, 17, 21, 23, 25, 32, 34, 38, 39, 43, 44, 52, 55, 61, 63, 64, 66, 68, 69, 71, 72, 73],
    "8": [1, 2, 3, 6, 11, 16, 17, 18, 20, 22, 23, 24, 25, 27, 30, 32, 37, 39, 45, 46, 49, 51, 52, 59, 64, 65, 71, 72],
}


def save_dataset_images(path=None, dataset: Dataset=None, train_validate=None, data_dict=None, tasks_nums=None):
    if dataset is None:
        print("Dataset not provided.")
        return
    if path is None:
        path = f"/home/dcorredor/github/Proyecto-ParkinsonDisease/NEW/dataset_images"
    os.makedirs(path, exist_ok=True)

    tasks_str = "".join(str(t) for t in tasks_nums) if tasks_nums else "none"

    print(f" ----------------***------------------- TAREAS SELECCIONADAS: {tasks_str}")

    final_path = os.path.join(path, tasks_str)
    os.makedirs(final_path, exist_ok=True)

    images_filename = {}

    for i in range(len(dataset)):
        patches, label, real_id, idx, t_number = dataset[i]

        patient = data_dict[real_id]

        img = patient.getTaskByTypeAndNum(RepresentationType.ENHANCED_STROKE, t_number).data

        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] == 1:
                img = img.squeeze(0)
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)

        filename = os.path.join(final_path, f"{train_validate}_img_idx{i:04d}_label_{label}_id{real_id}_task{t_number}.png")
        images_filename[idx] = filename
        
        cv2.imwrite(filename, img_uint8)
    return images_filename

def generate_analysis_csv(preds, targets, filenames, confidences, path="analysis2", tasks_nums=None, train_val="train", model=None, idx_list=[]):
    """
    Creates a CSV with train and validate results
    filename, target, predict, confidence
    """
    if not (len(preds) == len(targets) == len(filenames) == len(confidences)):
        raise ValueError("List parameters have different length")
    
    date = datetime.now()

    os.makedirs(path, exist_ok=True)

    tasks_str = "".join(str(t) for t in tasks_nums) if tasks_nums else "none"

    task_exec_path = os.path.join(path, tasks_str)
    os.makedirs(task_exec_path, exist_ok=True)

    filename = f"tasks_{tasks_str}_{model}_{train_val}_{date}.csv"
    full_path = os.path.join(task_exec_path, filename)
    with open(full_path, mode="w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        
        # Escribimos las filas combinando los elementos de las tres listas
        escritor.writerow(["filename", "prediction", "target", "park_neur confidence"])
        #for a, b, c, d in zip(filenames.values(), preds, targets, confidences):
        #    escritor.writerow([a, b, c, d])
        for idx, pred, target, conf in zip(idx_list, preds, targets, confidences):
            # filenames[idx] debe devolver el nombre de archivo correspondiente al índice
            fname = filenames[idx]  
            escritor.writerow([fname, pred, target, conf])
    
    print(f"Archivo '{full_path}' creado con éxito.")
    


def run_pipeline(train_data, validate_data, args=None, device=None, train_kargs=None, validate_kargs=None, writer=None, task_nums=[2]):
    train_dataset = PahawOfflineSimWindowDataset(train_data, None, 100, 50, task_nums, RepresentationType.ENHANCED_STROKE, "binary", augment=True)
    val_dataset = PahawOfflineSimWindowDataset(validate_data, None, 100, 50, task_nums, RepresentationType.ENHANCED_STROKE, "binary", augment=False)

#    patches_tensor, label, _, _ = train_dataset[0]
#    print(f"SHAPE: {patches_tensor.shape}")
#
    train_filenames = save_dataset_images(dataset=train_dataset, train_validate="train", data_dict=train_data, path="dataset_images", tasks_nums=task_nums)
    val_filenames = save_dataset_images(dataset=val_dataset, train_validate="validate", data_dict=validate_data, path="dataset_images", tasks_nums=task_nums)

    train_loader = DataLoader(train_dataset, **train_kargs)
    val_loader = DataLoader(val_dataset, **validate_kargs)

    early_stopping = EarlyStopping(patience=50, min_delta=1e-4)

    train_history = {
        "epoch": [],
        "loss": [],
        "accuracy": []
    }

    validate_history = {
        "epoch": [],
        "loss": [],
        "accuracy": []
    }


    model = cnnlstm.OfflineCnnLstm().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # baseline
    val_loss, val_acc, *_ = cnnlstm.validate(model, device, val_loader)
    train_history["loss"].append(np.nan)
    train_history["accuracy"].append(np.nan)
    train_history["epoch"].append(0)
    validate_history["loss"].append(val_loss)
    validate_history["accuracy"].append(val_acc)
    validate_history["epoch"].append(0)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_preds, train_targets, train_probs, train_idx, train_tasks = cnnlstm.train(args, model, device, train_loader, optimizer, epoch)

        val_loss, val_acc, val_preds, val_targets, val_probs, val_idx, val_tasks = cnnlstm.validate(model, device, val_loader)

        train_history["epoch"].append(epoch)
        train_history["loss"].append(train_loss)
        train_history["accuracy"].append(train_acc)

        validate_history["epoch"].append(epoch)
        validate_history["loss"].append(val_loss)
        validate_history["accuracy"].append(val_acc)

        early_stopping.step(val_loss, model)
        if early_stopping.stop:
            print(f"Early stopping at epoch {epoch}")
            model.load_state_dict(early_stopping.best_state)
            break

        scheduler.step()

#    errores = [idx for idx, (p, t) in enumerate(zip(train_preds, train_targets)) if p != t]
#
#    print("Fallos en índices:", errores)

    generate_analysis_csv(preds=train_preds, targets=train_targets, filenames=train_filenames, confidences=train_probs, train_val="train", model="CnnOnly", idx_list=train_idx, tasks_nums=task_nums)
    generate_analysis_csv(preds=val_preds, targets=val_targets, filenames=val_filenames, confidences=val_probs, train_val="validate", model="CnnOnly", idx_list=val_idx, tasks_nums=task_nums)

    return model, train_history, validate_history
