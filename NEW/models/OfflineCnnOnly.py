import torch
import torch.nn as nn
import torch.nn.functional as F

class OfflineCnnOnly(nn.Module):
    def __init__(self, feature_dim=128, num_classes=2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((5, 5)),
        )

        self.cnn_proj = nn.Linear(256 * 5 * 5, feature_dim)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, num_classes)
        )

    
    def forward(self, x):
        # Si ya no hay dimensión temporal
        if x.ndim == 4:  # (B, 1, H, W)
            x = self.cnn(x).view(x.size(0), -1)
            x = self.cnn_proj(x)
            return self.fc(x)
        else:
            # Si accidentalmente llega (B, T, 1, H, W)
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            x = self.cnn(x).view(B * T, -1)
            x = self.cnn_proj(x)
            x = x.view(B, T, -1).mean(dim=1)
            return self.fc(x)


def train(args, model, device, train_loader, optimizer, epoch, train_lossess, train_counter):
    model.train()

    all_predictions = []
    all_targets = []
    all_pd_neur_probs = []
    all_idx = []

    correct = 0

    for batch_idx, (data, target, _, idx) in enumerate(train_loader):
        # data shape: (B, T, 1, H, W), target shape: (B,)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #prediction
        pred = output.argmax(dim=1, keepdim=True)

        probs = F.softmax(output, dim=1)
        #confidences = probs.max(dim=1)[0]  # <---- AQUÍ SE CALCULA LA CONFIANZA
        all_pd_neur_probs.extend(probs[:, 1].detach().cpu().numpy())

        all_predictions.extend(pred.view(-1).cpu().numpy())
        all_targets.extend(target.cpu().numpy())

        all_idx.extend(idx.cpu().numpy())

        loss = F.cross_entropy(output, target, reduction='mean')
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_lossess.append(loss.item())
            train_counter.append((batch_idx * len(data)) + ((epoch - 1) * len(train_loader.dataset)))
            if args.dry_run:
                break
        correct += pred.eq(target.view_as(pred)).sum().item()
    print(f"Trained Epoch: {epoch}")
    print('\nTrain Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    return all_predictions, all_targets, all_pd_neur_probs, all_idx
    
def validate(model, device, validate_loader, validate_losses):
    model.eval()
    validate_loss = 0
    correct = 0

    all_predictions = []
    all_targets = []
    all_idx = []
    all_pd_neur_probs = []
    
    with torch.no_grad():
        for batch_idx, (data, target, _, idx) in enumerate(validate_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            validate_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get index of max log-probability
            probs = F.softmax(output, dim=1)
            #confidences = probs.max(dim=1)[0]  # <---- AQUÍ SE CALCULA LA CONFIANZA
            all_pd_neur_probs.extend(probs[:, 1].detach().cpu().numpy())

            all_predictions.extend(pred.view(-1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            start = batch_idx * validate_loader.batch_size
            all_idx.extend(idx.cpu().numpy())

            correct += pred.eq(target.view_as(pred)).sum().item()

    validate_loss /= len(validate_loader.dataset)
    validate_losses.append(validate_loss)

    print('\nValidate set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validate_loss, correct, len(validate_loader.dataset),
        100. * correct / len(validate_loader.dataset)))

    accuracy = 100. * correct / len(validate_loader.dataset)

    return all_predictions, all_targets, accuracy, all_pd_neur_probs, all_idx

