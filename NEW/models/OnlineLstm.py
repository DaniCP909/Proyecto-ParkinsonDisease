import torch
import torch.nn as nn
import torch.nn.functional as F

class OnlineLstm(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, num_layers=2, output_dim=2, dropout=0.3):
        super(OnlineLstm, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False  # opcional
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)  # Número de clases
        )

    def forward(self, x, lengths=None):
        if lengths is not None:
            packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_output, (hn, cn) = self.lstm(packed_input)
        else:
            _, (hn, cn) = self.lstm(x)

        # hn: [num_layers, batch, hidden_dim]
        out = hn[-1]  # Última capa oculta

        return self.fc(out)
    
def train(args, model, device, train_loader, optimizer, epoch, train_losses, train_counter):
    model.train()
    for batch_idx, (coords, labels) in enumerate(train_loader):
        coords, labels = coords.to(device), labels.to(device)  # coords shape: [B, T, 7], labels: [B]

        optimizer.zero_grad()
        output = model(coords)  # output: [B, num_classes]

        loss = F.cross_entropy(output, labels, reduction='mean')
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(coords)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            train_losses.append(loss.item())
            train_counter.append((batch_idx * len(coords)) + ((epoch - 1) * len(train_loader.dataset)))

            if args.dry_run:
                break

    print(f"Finished training epoch {epoch}")
    
def validate(model, device, validate_loader, validate_losses):
    model.eval()
    validate_loss = 0
    correct = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for coords, labels in validate_loader:
            coords, labels = coords.to(device), labels.to(device)

            output = model(coords)
            validate_loss += F.cross_entropy(output, labels, reduction='mean').item()

            preds = output.argmax(dim=1, keepdim=True)
            correct += preds.eq(labels.view_as(preds)).sum().item()

            all_predictions.extend(preds.view(-1).cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    validate_loss /= len(validate_loader.dataset)
    validate_losses.append(validate_loss)

    accuracy = 100. * correct / len(validate_loader.dataset)

    print(f"\nValidate set: Average loss: {validate_loss:.4f}, "
          f"Accuracy: {correct}/{len(validate_loader.dataset)} ({accuracy:.0f}%)\n")

    return all_predictions, all_targets, accuracy
