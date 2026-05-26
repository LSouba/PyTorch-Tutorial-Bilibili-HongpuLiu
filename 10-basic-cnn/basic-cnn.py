import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

class BasicCNN(torch.nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.pool  = torch.nn.MaxPool2d(2)
        self.fc    = torch.nn.Linear(in_features=20 * 4 * 4, out_features=10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

def train(model, criterion, optimizer, train_loader, epoch, device):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader, 0):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        if batch_idx % 300 == 299:
            print(f'Epoch {epoch} | Batch {batch_idx} | Loss {running_loss / total_samples:.4f}')

    return running_loss / total_samples

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1) # dim=1 means the dimension of the output tensor
            total += targets.size(0) # size(0) means the number of samples in the batch
            correct += (predicted == targets).sum().item()
    print(f'Test acc = {correct / total:.3f}')
    return correct / total

def run_training(device, train_dataset, test_dataset, batch_size, num_epochs=10, num_workers=4):
    print(f'Using device: {device}')
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    model = BasicCNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(num_epochs):
        start = time.perf_counter()
        train_loss = train(model, criterion, optimizer, train_loader, epoch, device)
        test_acc = test(model, test_loader, device)
        epoch_time = time.perf_counter() - start

        print(
            f'[{device.type.upper()}] Epoch {epoch} | '
            f'loss = {train_loss:.4f} | acc = {test_acc:.3f} | time = {epoch_time:.2f}s'
        )

def main():
    batch_size = 64
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) # mean 0.1307, std 0.3081

    train_dataset = datasets.MNIST(root=f'{Path.home()}/dataset/mnist', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=f'{Path.home()}/dataset/mnist', train=False, download=True, transform=transform)

    num_epochs = 10
    num_workers = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_training(device, train_dataset, test_dataset, batch_size, num_epochs, num_workers)

if __name__ == '__main__':
    main()
