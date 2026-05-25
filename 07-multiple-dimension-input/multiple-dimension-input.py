import torch
import numpy as np

class MultipleDimensionInputModel(torch.nn.Module):
    def __init__(self):
        super(MultipleDimensionInputModel, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=8, out_features=6, bias=True)
        self.linear2 = torch.nn.Linear(in_features=6, out_features=4, bias=True)
        self.linear3 = torch.nn.Linear(in_features=4, out_features=1, bias=True)
        self.activation1 = torch.nn.ReLU()
        self.activation2 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activation1(self.linear1(x))
        x = self.activation1(self.linear2(x))
        # x = self.activation2(self.linear3(x))
        x = self.linear3(x)
        return x

def main():
    xy = np.loadtxt('/home/lsouba/dataset/diabetes/diabetes.csv.gz', delimiter=',', dtype=np.float32)
    x_data = torch.from_numpy(xy[:, :-1])   # except last column
    y_data = torch.from_numpy(xy[:, [-1]])  # only last column

    model = MultipleDimensionInputModel()
    # criterion = torch.nn.BCELoss(reduction='sum')
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20000):
        # forward pass
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        if epoch % 1000 == 0:
            with torch.no_grad():
                acc = ((torch.sigmoid(y_pred) >= 0.5) == y_data).float().mean().item()
            print(f'Epoch {epoch} | loss = {loss.item():.8f} | acc = {acc:.3f}')

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # update weights
        optimizer.step()

    with torch.no_grad():
        y_prob = torch.sigmoid(model(x_data))
        acc = ((y_prob >= 0.5) == y_data).float().mean().item()
        print(f'Test acc = {acc:.3f}')

    w3 = model.linear3.weight.data.squeeze()
    print(f'w3 = {", ".join(f"{w:.3f}" for w in w3.tolist())}, b3 = {model.linear3.bias.item():.3f}')

if __name__ == '__main__':
    main()
