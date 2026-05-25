import torch

def forward(x, w1, w2, b):
    return w1 * (x ** 2) + w2 * x + b

def loss(x, y, w1, w2, b):
    y_pred = forward(x, w1, w2, b)
    return (y_pred - y) ** 2

def main():
    x_data = [1.0, 2.0, 3.0]
    y_data = [2.0, 4.0, 6.0]

    w1 = torch.Tensor([1.0])
    w1.requires_grad = True   # need to compute gradient, set to True
    w2 = torch.Tensor([1.0])
    w2.requires_grad = True
    b = torch.Tensor([1.0])
    b.requires_grad = True

    print(f'Predict (before training): f(4) = {forward(4, w1, w2, b).item():.3f}')

    for epoch in range(10000):
        for x, y in zip(x_data, y_data):
            loss_val = loss(x, y, w1, w2, b)
            loss_val.backward()
            if epoch % 1000 == 0:
                print(f'\tEpoch {epoch} | x = {x:.3f}, y = {y:.3f}, w1 = {w1.item():.3f}, w2 = {w2.item():.3f}, b = {b.item():.3f}')
                print(f'\t\tloss = {loss_val.item():.8f}, grad = {w1.grad.item():.3f}, {w2.grad.item():.3f}, {b.grad.item():.3f}')
            w1.data = w1.data - 0.01 * w1.grad.data
            w2.data = w2.data - 0.01 * w2.grad.data
            b.data = b.data - 0.01 * b.grad.data

            # clear gradient
            w1.grad.data.zero_()
            w2.grad.data.zero_()
            b.grad.data.zero_()

    print(f'w1 = {w1.item():.3f}, w2 = {w2.item():.3f}, b = {b.item():.3f}, loss = {loss_val.item():.8f}')

    print(f'Predict (after training): f(4) = {forward(4, w1, w2, b).item():.3f}')

if __name__ == '__main__':
    main()
