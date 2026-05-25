import numpy as np
import matplotlib.pyplot as plt

def forward(x, w):
    return x * w

def loss(x, y, w):
    y_pred = forward(x, w)
    return (y_pred - y) ** 2

def gradient(x, y, w):
    return 2 * x * (x * w - y)

def main():
    x_data = [1.0, 2.0, 3.0]
    y_data = [2.0, 4.0, 6.0]
    w = 1.0

    w_vals = []
    grad_vals = []
    loss_vals = []

    print(f'Predict (before training): f(4) = {forward(4, w):.3f}')

    for epoch in range(100):
        for x, y in zip(x_data, y_data):
            grad_val = gradient(x, y, w)
            grad_vals.append(grad_val)

            w = w - 0.01 * grad_val
            print(f'Epoch {epoch} | x = {x:.3f}, y = {y:.3f}, w = {w:.3f}, grad_val = {grad_val:.3f}')

            loss_val = loss(x, y, w)
            print(f'Epoch {epoch} | loss = {loss_val:.8f}')

            w_vals.append(w)
            loss_vals.append(loss_val)

    print(f'Predict (after training): f(4) = {forward(4, w):.3f}')

    # 画图：SGD 每个 epoch 有 len(x_data) 次更新，x 轴用 step 索引
    steps = list(range(len(w_vals)))
    plt.figure(figsize=(10, 6))
    plt.plot(steps, w_vals, label='w', color='blue')
    plt.plot(steps, grad_vals, label='grad_val', color='red')
    plt.plot(steps, loss_vals, label='loss_val', color='green')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title('w, grad_val, loss_val vs Step (SGD)')
    plt.legend()
    plt.grid(True)
    plt.savefig('curves_sgd.png', dpi=150)
    print('已保存: curves_sgd.png')

if __name__ == '__main__':
    main()
