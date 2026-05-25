import numpy as np
import matplotlib.pyplot as plt

def forward(x, w, b):
    return x * w + b

def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2

def mse(x_data, y_data, w, b):
    total = sum(loss(x, y, w, b) for x, y in zip(x_data, y_data))
    return total / len(x_data)

def main():
    x_data = [1.0, 2.0, 3.0]
    y_data = [2.0, 4.0, 6.0]

    w_range = np.arange(0.0, 4.1, 0.1)
    b_range = np.arange(-2.0, 3.1, 0.1)
    W, B = np.meshgrid(w_range, b_range)
    Loss = np.zeros_like(W)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Loss[i, j] = mse(x_data, y_data, W[i, j], B[i, j])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(W, B, Loss, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_xlabel('w')
    ax.set_ylabel('b')
    ax.set_zlabel('Loss')
    ax.set_title('Loss vs w and b')
    fig.colorbar(surf, shrink=0.5, aspect=12, label='Loss')
    plt.savefig('loss_affine.png', dpi=150)
    print('已保存: loss_affine.png')

if __name__ == '__main__':
    main()
