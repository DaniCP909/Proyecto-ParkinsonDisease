import random
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import pahaw_loader
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

from datasets.PahawOfflineSimDataset import PahawOfflineSimDataset
from models.OfflineCnnLstm import OfflineCnnLstm, train, validate

from torch.utils.tensorboard import SummaryWriter

from time import time

from subset_utils import build_subsets, build_overfit_subsets
from pipeline import run_pipeline

task_number = 8

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

    args = parser.parse_args()
    writer = SummaryWriter("runs/pd-detection")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    print(f'*** Training settings: batch_size:{args.batch_size}, validate_batch_size:{args.validate_batch_size}, epochs:{args.epochs}, lr:{args.lr}, gamma:{args.gamma}, no-cuda:{args.no_cuda}, no_mps:{args.no_mps}, dry_run:{args.dry_run}, seed:{args.seed}, log_interval:{args.log_interval}, save_model:{args.save_model}')

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    validate_kargs = {'batch_size': args.validate_batch_size}
    if use_cuda:
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': False}
        train_kwargs.update(cuda_kwargs)
        validate_kargs.update(cuda_kwargs)
    
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    print(f"Device: {device}")

    t0_load_data = time()

    subjects_pd_status_years, subjects_tasks = pahaw_loader.load()

    #train_ids, train_label_img, validate_ids, validate_label_img = build_subsets(subjects_pd_status_years, subjects_tasks, args, task_number, task_number+1)
    train_ids, train_label_img, validate_ids, validate_label_img = build_subsets(subjects_pd_status_years, subjects_tasks, args, min_task=task_number, max_task=task_number+1)
    #print("------------Creacion de subsets:")
    #print(f"All IDs: {subjects_tasks.keys()}")
    print(f"TRAIN: {train_ids}")
    print(f"VALIDATE: {validate_ids}")

    elapsed_load_data = time() - t0_load_data
    print(f"PaHaW data loaded and patches generated in {(elapsed_load_data):.2f}s")

    t0_train = time()

    print(f"Longitud train {len(train_label_img)} Longitud validate {len(validate_label_img)}")

    model, accuracy_history, train_losses, validate_losses = run_pipeline(train_ids, validate_ids, train_label_img, validate_label_img, args, device, train_kwargs, validate_kargs, writer, task_number) 

    elapsed_train = time() - t0_train
    print(f"Model trained in {(elapsed_train):.2f}s")

    print(subjects_pd_status_years)

#    #Plot results
#    performance_fig = plt.figure()
#    plt.plot(train_counter, train_losses, color='green', zorder=3)
#    plt.scatter(validate_counter, validate_losses, color='purple', zorder=2)
#    plt.legend(['Train loss', 'Validate loss'], loc='upper right')
#    plt.xlabel('number of training samples seen')
#    plt.ylabel('negative log likelihood loss')
#    performance_fig.savefig(f"results/performance.png")
#
#    if args.save_model:
#        torch.save(model.state_dict(), f'results/model_results/my_model.pt')
#        torch.save(optimizer.state_dict(), f'results/model_results/my_optimizer.pt')


if __name__ == '__main__':
    main()