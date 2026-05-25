import numpy as np
import matplotlib.pyplot as plt

def forward(x, w):
    return x * w

def loss(x, y, w):
    y_pred = forward(x, w)
    return (y_pred - y) ** 2

def main():
    x_data = [1.0, 2.0, 3.0]
    y_data = [2.0, 4.0, 6.0]

    w_list = []
    mse_list = []

    for w in np.arange(0.0, 4.1, 0.1):
        loss_val = 0
        for x, y in zip(x_data, y_data):
            y_pred_val = forward(x, w)
            loss_val += loss(x, y, w)
        loss_val /= 3
        w_list.append(w)
        mse_list.append(loss_val)

    plt.plot(w_list, mse_list)
    plt.xlabel('w')
    plt.ylabel('Loss')
    plt.title('Loss vs w')
    # plt.show()
    plt.savefig('loss_linear.png', dpi=150)
    print('已保存: loss_linear.png')

if __name__ == '__main__':
    main()
