# pipeline.py
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from models.OfflineCnnLstm import OfflineCnnLstm, train, validate
from PahawOfflineSimDataset import PahawOfflineSimDataset

wrong_predicts = {
    "2": [5, 16, 17, 21, 24, 27, 31, 32, 35, 36, 38, 39, 46, 52, 56, 60, 64, 65, 70, 71, 72], #batch_size=2
    "3": [4, 5, 9, 19, 21, 27, 31, 32, 35, 36, 39, 42, 44, 47, 50, 56, 65, 71],
    "4": [1, 5, 7, 11, 12, 13, 17, 19, 23, 27, 34, 35, 40, 48, 55, 61, 62, 67],
    "5": [1, 2, 3, 4, 7, 8, 10, 15, 16, 18, 20, 23, 24, 25, 28, 29, 31, 33, 34, 37, 49, 50, 53, 56, 59, 60, 62, 63, 64, 66, 67, 69, 70, 72, 73],
    "6": [0, 2, 3, 5, 9, 10, 11, 13, 14, 16, 17, 19, 21, 22, 27, 29, 30, 32, 34, 36, 38, 40, 41, 42, 44, 45, 46, 47, 52, 54, 56, 65, 66, 68, 71, 72, 74],
    "7": [1, 2, 7, 8, 12, 14, 16, 17, 21, 23, 25, 32, 34, 38, 39, 43, 44, 52, 55, 61, 63, 64, 66, 68, 69, 71, 72, 73],
    "8": [1, 2, 3, 6, 11, 16, 17, 18, 20, 22, 23, 24, 25, 27, 30, 32, 37, 39, 45, 46, 49, 51, 52, 59, 64, 65, 71, 72],
}

def run_pipeline(train_ids, validate_ids, train_data, validate_data, args=None, device=None, train_kargs=None, validate_kargs=None, writer=None, task=2):
    print(f"ARGS: {args}")
    train_dataset = PahawOfflineSimDataset(train_data, device=device, patch_w=200, stepsize=30, id_list=train_ids)
    val_dataset = PahawOfflineSimDataset(validate_data, device=device, patch_w=200, stepsize=30, id_list=validate_ids)

    print(f"LEN train: {len(train_dataset)}")
    print(f"LEN validate: {len(val_dataset)}")
    filtered_train_dataset = Subset(train_dataset, [i for i in range(len(train_dataset)) if i not in wrong_predicts[f"{task}"]])
    filtered_validate_dataset = Subset(val_dataset, [i for i in range(len(val_dataset)) if i not in wrong_predicts[f"{task}"]])

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

    train_loader = DataLoader(filtered_train_dataset, **train_kargs)
    val_loader = DataLoader(filtered_validate_dataset, **validate_kargs)

    train_losses, validate_losses, accuracy_history = [], [], []
    train_counter, validate_counter = [], []

    model = OfflineCnnLstm().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # baseline
    _, _, all_indices, acc = validate(model, device, val_loader, validate_losses)
    accuracy_history.append(acc)
    validate_counter.append(0)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, train_losses, train_counter)
        preds, targets, all_indices,  acc = validate(model, device, val_loader, validate_losses)
        accuracy_history.append(acc)
        validate_counter.append(epoch * len(train_loader.dataset))
        if writer is not None:
            writer.add_scalar("Accuracy", acc, epoch)
        scheduler.step()

    errores = [idx for idx, (p, t) in enumerate(zip(preds, targets)) if p != t]

    print("Fallos en índices:", errores)

    return model, accuracy_history, train_losses, validate_losses
