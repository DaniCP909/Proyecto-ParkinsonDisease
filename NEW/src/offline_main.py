import random
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import random
import argparse
import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
from collections import Counter
import os
import numpy as np
import cv2

import pandas as pd

from datasets.PahawOfflineSimDataset import PahawOfflineSimDataset
from models.OfflineCnnLstm import OfflineCnnLstm, train, validate
from domain.PahawLoader import PahawLoader
from domain.RepresentationType import RepresentationType
from domain.PahawSplitter import PahawSplitter

from domain.Patient import Patient

from torch.utils.tensorboard import SummaryWriter

from time import time

from pipeline.pipeline import run_pipeline

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='PaHaW offline training')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default = 64)')
    parser.add_argument('--validate-batch-size', type=int, default=64, metavar='N', help='input batch size for validating (default = 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N', help='number of epochs to train (default = 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default = 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='learning rate step gamma (default = 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='desables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False, help='disables MacOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default = 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='Saves current Model')
    parser.add_argument('--segment', type=int, default=0, metavar='N', help='segment (default = 0)')
    parser.add_argument('--isolated', type=int, default=0, metavar='N', help='isolated patient (default = 0)')
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="+",
        default=[2],
        help="Task list to use",
    )

    args = parser.parse_args()
    writer = SummaryWriter("runs/pd-detection")

    task_numbers = args.tasks

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    set_seed(args.seed)

    print(f'*** Training settings: batch_size:{args.batch_size}, validate_batch_size:{args.validate_batch_size}, epochs:{args.epochs}, lr:{args.lr}, gamma:{args.gamma}, no-cuda:{args.no_cuda}, no_mps:{args.no_mps}, dry_run:{args.dry_run}, seed:{args.seed}, log_interval:{args.log_interval}, save_model:{args.save_model}')

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': True
    }
    
    validate_kwargs = {
        'batch_size': args.validate_batch_size,
        'shuffle': False
    }
    
    if use_cuda:
        train_kwargs.update({
            'num_workers': 2,
            'pin_memory': True
        })
        validate_kwargs.update({
            'num_workers': 2,
            'pin_memory': True
        })
    
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    print(f"Device: {device}")

    results_dir = "results_singletask_isolated"
    os.makedirs(results_dir, exist_ok=True)
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    csv_dir = os.path.join(results_dir, "history_csvs")
    os.makedirs(csv_dir, exist_ok=True)

    t0_load_data = time()

    pahaw_loader = PahawLoader()
    patients_dict = pahaw_loader.load()

    all_ids = list(patients_dict.keys())
    train_list = all_ids[:args.isolated] + all_ids[args.isolated+1:]
    test_list = [all_ids[args.isolated]]

    elapsed_load_data = time() - t0_load_data
    print(f"PaHaW data loaded and patches generated in {(elapsed_load_data):.2f}s")


    splitter = PahawSplitter(patients_dict)
    train, val = splitter.custom_split(train_list, test_list)

    print(train)

    t0_train = time()

    model, train_history, validate_history = run_pipeline(train, val, args, device, train_kwargs, validate_kwargs, task_nums=task_numbers)

    elapsed_train = time() - t0_train
    print(f"Model trained in {(elapsed_train):.2f}s")

    model_path = os.path.join(models_dir, f"best_model_patient{args.isolated}_tasks{'_'.join(map(str, task_numbers))}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    train_csv_path = os.path.join(csv_dir, f"train_history_patient{args.isolated}_tasks{'_'.join(map(str, task_numbers))}.csv")
    pd.DataFrame(train_history).to_csv(train_csv_path, index=False)
    print(f"Train history saved to {train_csv_path}")

    # Guardar historial de validación
    val_csv_path = os.path.join(csv_dir, f"validate_history_patient{args.isolated}_tasks{'_'.join(map(str, task_numbers))}.csv")
    pd.DataFrame(validate_history).to_csv(val_csv_path, index=False)
    print(f"Validate history saved to {val_csv_path}")

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(train_history["epoch"], train_history["loss"], label="Train Loss")
    plt.plot(validate_history["epoch"], validate_history["loss"], label="Validate Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(results_dir, f"loss_plot_patient{args.isolated}_tasks{'_'.join(map(str, task_numbers))}.png"))
    plt.close()

    plt.figure()
    plt.plot(train_history["epoch"], train_history["accuracy"], label="Train Accuracy")
    plt.plot(validate_history["epoch"], validate_history["accuracy"], label="Validate Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(results_dir, f"accuracy_plot_patient{args.isolated}_tasks{'_'.join(map(str, task_numbers))}.png"))
    plt.close()

#    if args.save_model:
#        torch.save(model.state_dict(), f'results/model_results/my_model.pt')
#        torch.save(optimizer.state_dict(), f'results/model_results/my_optimizer.pt')


if __name__ == '__main__':
    main()