# pipeline.py
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
#from models.OfflineCnnLstm import OfflineCnnLstm, train, validate
from models.OfflineCnnOnly import OfflineCnnOnly, train, validate
from datasets.PahawOfflineSimDataset import PahawOfflineSimDataset
from datasets.PahawOfflineSimWindowDataset import PahawOfflineSimWindowDataset
import os
import cv2
import numpy as np
import csv
from datetime import datetime

from domain.RepresentationType import RepresentationType

wrong_predicts = {
    "2": [5, 16, 17, 21, 24, 27, 31, 32, 35, 36, 38, 39, 46, 52, 56, 60, 64, 65, 70, 71, 72], #batch_size=2
    "3": [4, 5, 9, 19, 21, 27, 31, 32, 35, 36, 39, 42, 44, 47, 50, 56, 65, 71],
    "4": [1, 5, 7, 11, 12, 13, 17, 19, 23, 27, 34, 35, 40, 48, 55, 61, 62, 67],
    "5": [1, 2, 3, 4, 7, 8, 10, 15, 16, 18, 20, 23, 24, 25, 28, 29, 31, 33, 34, 37, 49, 50, 53, 56, 59, 60, 62, 63, 64, 66, 67, 69, 70, 72, 73],
    "6": [0, 2, 3, 5, 9, 10, 11, 13, 14, 16, 17, 19, 21, 22, 27, 29, 30, 32, 34, 36, 38, 40, 41, 42, 44, 45, 46, 47, 52, 54, 56, 65, 66, 68, 71, 72, 74],
    "7": [1, 2, 7, 8, 12, 14, 16, 17, 21, 23, 25, 32, 34, 38, 39, 43, 44, 52, 55, 61, 63, 64, 66, 68, 69, 71, 72, 73],
    "8": [1, 2, 3, 6, 11, 16, 17, 18, 20, 22, 23, 24, 25, 27, 30, 32, 37, 39, 45, 46, 49, 51, 52, 59, 64, 65, 71, 72],
}


def save_dataset_images(path=None, dataset: PahawOfflineSimDataset=None, train_validate=None, task_num=2, window=False):
    if dataset is None:
        print("Dataset not provided.")
        return
    if path is None:
        path = f"/home/dcorredor/github/Proyecto-ParkinsonDisease/NEW/dataset_images/{task_num}"
    os.makedirs(path, exist_ok=True)


    images_filename = {}

    for i in range(len(dataset)):
        if not window:
            img, label, real_id, idx = dataset[i]

            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            if img.ndim == 3 and img.shape[0] == 1:
                img = img.squeeze(0)
            img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)

        filename = os.path.join(path, f"{train_validate}_img_idx{i:04d}_label_{label}_id{real_id}.png")
        images_filename[idx] = filename
        if not window:
            cv2.imwrite(filename, img_uint8)
    return images_filename

def generate_analysis_csv(preds, targets, filenames, confidences, path="analysis", task_num=None, train_val="train", model=None, idx_list=[]):
    """
    Creates a CSV with train and validate results
    filename, target, predict, confidence
    """
    if not (len(preds) == len(targets) == len(filenames) == len(confidences)):
        raise ValueError("List parameters have different length")
    
    date = datetime.now()
    task_path = os.path.join(path, f"{task_num}")

    os.makedirs(task_path, exist_ok=True)

    filename = f"task{task_num}_{model}_{train_val}_{date}.csv"
    full_path = os.path.join(task_path, filename)
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
    train_dataset = PahawOfflineSimDataset(train_data, None, 200, 2, task_nums, RepresentationType.SIMPLE_STROKE, "binary")
    val_dataset = PahawOfflineSimDataset(validate_data, None, 200, 2, task_nums, RepresentationType.SIMPLE_STROKE, "binary")

#    patches_tensor, label, _, _ = train_dataset[0]
#    print(f"SHAPE: {patches_tensor.shape}")
#
#    train_filenames = save_dataset_images(dataset=train_dataset, train_validate="train", task_num=task, window=False)
#    val_filenames = save_dataset_images(dataset=val_dataset, train_validate="validate", task_num=task, window=False)
#
#    print(f"LEN train: {len(train_dataset)}")
#    print(f"LEN validate: {len(val_dataset)}")
#    filtered_train_dataset = Subset(train_dataset, [i for i in range(len(train_dataset)) if i not in wrong_predicts[f"{task}"]])
#    filtered_validate_dataset = Subset(val_dataset, [i for i in range(len(val_dataset)) if i not in wrong_predicts[f"{task}"]])

#    # Comprobación de igualdad entre train y validate (overfit dataset)
#    train_items = [(task, label) for task, label in train_data]  # tu lista original
#    validate_items = [(task, label) for task, label in validate_data]  # o validate_data_label_img si es diferente
#
#    # Comparar longitud
#    print("Longitudes iguales?", len(train_items) == len(validate_items))
#
#    # Comparar labels
#    train_labels = [label for _, label in train_items]
#    validate_labels = [label for _, label in validate_items]
#    print("Labels iguales?", train_labels == validate_labels)
#
#    # Comparar IDs (si quieres asegurar que las tareas son las mismas)
#    train_ids = [id(task) for task, _ in train_items]
#    validate_ids = [id(task) for task, _ in validate_items]
#    print("IDs de tareas iguales?", train_ids == validate_ids)

    train_loader = DataLoader(train_dataset, **train_kargs)
    val_loader = DataLoader(val_dataset, **validate_kargs)

    train_losses, validate_losses, accuracy_history = [], [], []
    train_counter, validate_counter = [], []

    model = OfflineCnnOnly().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # baseline
    _, _, acc, _, val_idxs = validate(model, device, val_loader, validate_losses)
    accuracy_history.append(acc)
    validate_counter.append(0)

    for epoch in range(1, args.epochs + 1):
        train_preds, train_targets, train_confidences, train_idxs = train(args, model, device, train_loader, optimizer, epoch, train_losses, train_counter)
        val_preds, val_targets,  acc, val_confidences, val_idxs = validate(model, device, val_loader, validate_losses)
        accuracy_history.append(acc)
        validate_counter.append(epoch * len(train_loader.dataset))
        if writer is not None:
            writer.add_scalar("Accuracy", acc, epoch)
        scheduler.step()

#    errores = [idx for idx, (p, t) in enumerate(zip(train_preds, train_targets)) if p != t]
#
#    print("Fallos en índices:", errores)

#    generate_analysis_csv(preds=train_preds, targets=train_targets, filenames=train_filenames, confidences=train_confidences, task_num=task, train_val="train", model="CnnOnly", idx_list=train_idxs)
#    generate_analysis_csv(preds=val_preds, targets=val_targets, filenames=val_filenames, confidences=val_confidences, task_num=task, train_val="validate", model="CnnOnly", idx_list=val_idxs)

    return model, accuracy_history, train_losses, validate_losses
