import numpy as np
import matplotlib.pyplot as plt

def forward(x, w):
    return x * w

def cost(x, y, w):
    cost = 0
    for xi, yi in zip(x, y):
        y_pred = forward(xi, w)
        cost += (y_pred - yi) ** 2
    return cost / len(x)

def gradient(x, y, w):
    grad = 0
    for xi, yi in zip(x, y):
        grad += 2 * xi * (xi * w - yi)
    return grad / len(x)

def main():
    x_data = [1.0, 2.0, 3.0]
    y_data = [2.0, 4.0, 6.0]
    w = 1.0

    w_vals = []
    grad_vals = []
    cost_vals = []

    print(f'Predict (before training): f(4) = {forward(4, w):.3f}')

    for epoch in range(100):
        cost_val = cost(x_data, y_data, w)
        grad_val = gradient(x_data, y_data, w)

        w_vals.append(w)
        grad_vals.append(grad_val)
        cost_vals.append(cost_val)

        w = w - 0.01 * grad_val
        print(f'Epoch {epoch} | w = {w:.3f}, grad_val = {grad_val:.3f}, cost = {cost_val:.8f}')

    print(f'Predict (after training): f(4) = {forward(4, w):.3f}')

    # 画图，x轴是epoch 0~100, y轴是数值，输出3个变量的曲线（w, grad_val, cost_val），用不同颜色

    epochs = list(range(100))
    plt.figure(figsize=(10,6))
    plt.plot(epochs, w_vals, label='w', color='blue')
    plt.plot(epochs, grad_vals, label='grad_val', color='red')
    plt.plot(epochs, cost_vals, label='cost_val', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('w, grad_val, cost_val vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('curves_gd.png', dpi=150)
    print('已保存: curves_gd.png')

if __name__ == '__main__':
    main()
