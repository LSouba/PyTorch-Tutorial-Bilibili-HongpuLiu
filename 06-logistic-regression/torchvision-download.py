import torchvision

# train_dataset = torchvision.datasets.MNIST(root='/home/lsouba/dataset/mnist', train=True, download=True, transform=None)
# test_dataset = torchvision.datasets.MNIST(root='/home/lsouba/dataset/mnist', train=False, download=True, transform=None)

train_dataset = torchvision.datasets.CIFAR10(root='/home/lsouba/dataset/cifar10', train=True, download=True, transform=None)
test_dataset = torchvision.datasets.CIFAR10(root='/home/lsouba/dataset/cifar10', train=False, download=True, transform=None)
