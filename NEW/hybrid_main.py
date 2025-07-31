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
from collections import Counter

from models.OnlineLstm import OnlineLstm, train, validate

from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

from time import time

def collate_fn_pad(batch):
    coords_batch, labels_batch = zip(*batch)
    coords_padded = pad_sequence(
        coords_batch, batch_first=True, padding_value=0.0
    )
    labels = torch.tensor(labels_batch, dtype=torch.long)
    return coords_padded, labels

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
            'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        validate_kargs.update(cuda_kwargs)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1317,), (0.3081,))
    ])

    print(f"Device: {device}")

    t0_load_data = time()

    subjects_pd_status_years, subjects_tasks = pahaw_loader.load()

    h_id_list = []
    pd_id_list = []
    subjects_ids = list(subjects_tasks.keys())

    for subject_id in subjects_ids:
        if subjects_pd_status_years[subject_id][0] == 0:
            h_id_list.append(subject_id)
        else:
            pd_id_list.append(subject_id)
    random.Random(args.seed).shuffle(h_id_list)
    random.Random(args.seed).shuffle(pd_id_list)

    validate_id_list = h_id_list[:7]
    train_id_list = h_id_list[7:]
    validate_id_list.extend(pd_id_list[:6])
    train_id_list.extend(pd_id_list[6:])

    validate_id_list.sort()
    train_id_list.sort()

    train_id_set = set(train_id_list)
    validate_id_set = set(validate_id_list)

    elapsed_load_data = time() - t0_load_data
    print(f"PaHaW data loaded and patches generated in {(elapsed_load_data):.2f}s")

    validate_data_label_img = []
    train_data_label_img = []
    for subject_id in subjects_ids:
        for task_number in range(2, 9):
            task = subjects_tasks[subject_id].get(task_number)
            if task is not None:
                print(task.getCanavases())
            break
        break

if __name__ == '__main__':
    main()
