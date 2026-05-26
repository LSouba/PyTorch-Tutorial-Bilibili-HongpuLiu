import torchvision
from pathlib import Path

# train_dataset = torchvision.datasets.MNIST(root=f'{Path.home()}/dataset/mnist', train=True, download=True, transform=None)
# test_dataset = torchvision.datasets.MNIST(root=f'{Path.home()}/dataset/mnist', train=False, download=True, transform=None)

train_dataset = torchvision.datasets.CIFAR10(root=f'{Path.home()}/dataset/cifar10', train=True, download=True, transform=None)
test_dataset = torchvision.datasets.CIFAR10(root=f'{Path.home()}/dataset/cifar10', train=False, download=True, transform=None)
