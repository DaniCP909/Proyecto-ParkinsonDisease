import torch
import torch.nn as nn
import torch.nn.functional as F

class OfflineCnnLstm(nn.Module):
    def __init__(self, feature_dim=32, lstm_hidden=64, num_classes=2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5)),
        )

        self.cnn_proj = nn.Linear(256 * 5 * 5, feature_dim)
        self.feature_dim = feature_dim 

        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x).view(B * T, -1)
        x = self.cnn_proj(x)
        x = x.view(B, T, -1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n.squeeze(0))

def train(args, model, device, train_loader, optimizer, epoch, train_lossess, train_counter):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data shape: (B, T, 1, H, W), target shape: (B,)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
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
    print(f"Trained Epoch: {epoch}")
    
def validate(model, device, validate_loader, validate_losses):
    model.eval()
    validate_loss = 0
    correct = 0

    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in validate_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            validate_loss += F.cross_entropy(output, target, reduction='mean').item()
            pred = output.argmax(dim=1, keepdim=True)  # get index of max log-probability

            all_predictions.extend(pred.view(-1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            correct += pred.eq(target.view_as(pred)).sum().item()

    validate_loss /= len(validate_loader.dataset)
    validate_losses.append(validate_loss)

    print('\nValidate set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validate_loss, correct, len(validate_loader.dataset),
        100. * correct / len(validate_loader.dataset)))

    accuracy = 100. * correct / len(validate_loader.dataset)

    return all_predictions, all_targets, accuracy

