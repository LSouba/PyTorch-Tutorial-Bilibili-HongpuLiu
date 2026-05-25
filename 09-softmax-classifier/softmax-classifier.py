import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class SoftmaxClassifier(torch.nn.Module):
    def __init__(self):
        super(SoftmaxClassifier, self).__init__()

        # 784 = 28 * 28
        self.linear1 = torch.nn.Linear(in_features=784, out_features=512, bias=True)
        self.linear2 = torch.nn.Linear(in_features=512, out_features=256, bias=True)
        self.linear3 = torch.nn.Linear(in_features=256, out_features=128, bias=True)
        self.linear4 = torch.nn.Linear(in_features=128, out_features=64, bias=True)
        self.linear5 = torch.nn.Linear(in_features=64, out_features=10, bias=True)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)
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

    model = SoftmaxClassifier().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    history = {'train_loss': [], 'test_acc': [], 'epoch_time': []}
    for epoch in range(num_epochs):
        start = time.perf_counter()
        train_loss = train(model, criterion, optimizer, train_loader, epoch, device)
        test_acc = test(model, test_loader, device)
        epoch_time = time.perf_counter() - start

        history['train_loss'].append(train_loss)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)
        print(
            f'[{device.type.upper()}] Epoch {epoch} | '
            f'loss = {train_loss:.4f} | acc = {test_acc:.3f} | time = {epoch_time:.2f}s'
        )

    history['total_time'] = sum(history['epoch_time'])
    return history

def plot_device_comparison(cuda_history, cpu_history, output_path):
    num_epochs = len(cpu_history['test_acc'])
    epochs = range(1, num_epochs + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    if cuda_history is not None:
        ax.plot(epochs, cuda_history['test_acc'], 'o-', label='CUDA', color='tab:orange')
    ax.plot(epochs, cpu_history['test_acc'], 's-', label='CPU', color='tab:blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy')
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    if cuda_history is not None:
        ax.plot(epochs, cuda_history['epoch_time'], 'o-', label='CUDA', color='tab:orange')
    ax.plot(epochs, cpu_history['epoch_time'], 's-', label='CPU', color='tab:blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Time per Epoch (s)')
    ax.set_title('Epoch Training Time')
    ax.legend()
    ax.grid(True)

    ax = axes[2]
    if cuda_history is not None:
        ax.plot(epochs, cuda_history['train_loss'], 'o-', label='CUDA', color='tab:orange')
    ax.plot(epochs, cpu_history['train_loss'], 's-', label='CPU', color='tab:blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True)

    if cuda_history is not None:
        speedup = cpu_history['total_time'] / cuda_history['total_time']
        fig.suptitle(
            f'CUDA vs CPU Performance '
            f'(total: CUDA {cuda_history["total_time"]:.1f}s, '
            f'CPU {cpu_history["total_time"]:.1f}s, speedup {speedup:.2f}x)',
            fontsize=12,
        )
    else:
        fig.suptitle(f'CPU Performance (total: {cpu_history["total_time"]:.1f}s)', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=150)
    print(f'已保存: {output_path}')

def main():
    batch_size = 64
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) # mean 0.1307, std 0.3081

    train_dataset = datasets.MNIST(root='/home/lsouba/dataset/mnist', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='/home/lsouba/dataset/mnist', train=False, download=True, transform=transform)

    num_epochs = 10
    num_workers = 4
    cuda_history = None
    if torch.cuda.is_available():
        cuda_history = run_training(
            torch.device('cuda'), train_dataset, test_dataset, batch_size, num_epochs, num_workers
        )

    cpu_history = run_training(
        torch.device('cpu'), train_dataset, test_dataset, batch_size, num_epochs, num_workers
    )

    output_path = Path(__file__).with_name('device_comparison.png')
    plot_device_comparison(cuda_history, cpu_history, output_path)


if __name__ == '__main__':
    main()
