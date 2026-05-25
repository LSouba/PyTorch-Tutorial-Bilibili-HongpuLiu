import torch

def forward(x, w):
    return x * w

def loss(x, y, w):
    y_pred = forward(x, w)
    return (y_pred - y) ** 2

def main():
    x_data = [1.0, 2.0, 3.0]
    y_data = [2.0, 4.0, 6.0]

    w = torch.Tensor([1.0])
    w.requires_grad = True   # need to compute gradient, set to True

    print(f'Predict (before training): f(4) = {forward(4, w).item():.3f}')

    for epoch in range(100):
        for x, y in zip(x_data, y_data):
            loss_val = loss(x, y, w)
            loss_val.backward()
            print(f'\tEpoch {epoch} | x = {x:.3f}, y = {y:.3f}, w = {w.item():.3f}, loss = {loss_val.item():.8f}, grad = {w.grad.item():.3f}')
            w.data = w.data - 0.01 * w.grad.data

            # clear gradient
            w.grad.data.zero_()

        print(f'Epoch {epoch} | loss = {loss_val.item():.8f}')

    print(f'Predict (after training): f(4) = {forward(4, w).item():.3f}')

if __name__ == '__main__':
    main()
