import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class OfflineResnetCnnLstm(nn.Module):
    def __init__(self, feature_dim=512, lstm_hidden=256, num_classes=2, pretrained=True):
        super().__init__()

        # 1) Backbone ResNet18
        backbone = resnet18(pretrained=pretrained)

        # Cambiamos la primera conv para aceptar 1 canal
        # (resnet original espera 3 canales)
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Quitamos la FC final
        self.cnn = nn.Sequential(
            *list(backbone.children())[:-1]
        )
        # Ahora self.cnn da salida: (B, 512, 1, 1)

        self.feature_dim = feature_dim  # 512 por defecto

        # 2) LSTM sobre secuencia de patches
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
            bidirectional=False
        )

        # 3) Clasificador final
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(lstm_hidden, num_classes)
        )


    def forward(self, x):
        # x: (B, T, 1, H, W)

        # modo single image
        #if x.ndim == 4:
        #    x = self.cnn(x).view(x.size(0), -1)
        #    return self.fc(x)

        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)

        # CNN: (B*T, 512)
        x = self.cnn(x).view(B*T, -1)

        # secuencia: (B, T, 512)
        x = x.view(B, T, self.feature_dim)

        out, (h_n, c_n) = self.lstm(x)

        last_hidden = h_n[-1]
        return self.fc(last_hidden)



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    # Listas que quieres mantener para monitorización
    all_predictions = []
    all_targets = []
    all_pd_neur_probs = []
    all_idx = []
    tasks_nums = []

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target, _, idx, t_number) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # Predicciones y probabilidades
        pred = output.argmax(dim=1, keepdim=True)
        probs = F.softmax(output, dim=1)
        all_pd_neur_probs.extend(probs[:, 1].detach().cpu().numpy())
        all_predictions.extend(pred.view(-1).cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        all_idx.extend(idx.cpu().numpy())
        if isinstance(t_number, torch.Tensor):
            tasks_nums.extend([int(x) for x in t_number.view(-1)])
        else:
            tasks_nums.append(int(t_number))

        # Loss por batch
        loss = F.cross_entropy(output, target, reduction='sum')
        loss.backward()
        optimizer.step()

        # Acumulamos loss y aciertos
        total_loss += loss.item()
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        # Log opcional
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / target.size(0)))
            if args.dry_run:
                break

    avg_loss = total_loss / total
    accuracy = correct / total

    print(f"Trained Epoch: {epoch}")
    print('Train Accuracy: {}/{} ({:.0f}%)\n'.format(correct, total, 100. * accuracy))

    return avg_loss, accuracy, all_predictions, all_targets, all_pd_neur_probs, all_idx, tasks_nums

    
def validate(model, device, validate_loader):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_targets = []
    all_idx = []
    all_pd_neur_probs = []
    tasks_nums = []

    with torch.no_grad():
        for batch_idx, (data, target, _, idx, t_number) in enumerate(validate_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            pred = output.argmax(dim=1, keepdim=True)
            probs = F.softmax(output, dim=1)
            all_pd_neur_probs.extend(probs[:, 1].detach().cpu().numpy())
            all_predictions.extend(pred.view(-1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_idx.extend(idx.cpu().numpy())
            if isinstance(t_number, torch.Tensor):
                tasks_nums.extend([int(x) for x in t_number.view(-1)])
            else:
                tasks_nums.append(int(t_number))

            total_loss += F.cross_entropy(output, target, reduction='sum').item()
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    print('Validate set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, total, 100. * accuracy))

    return avg_loss, accuracy, all_predictions, all_targets, all_pd_neur_probs, all_idx, tasks_nums

