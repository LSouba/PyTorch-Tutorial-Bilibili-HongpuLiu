import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DiabetesDataset(Dataset):
    def __init__(self, file_path):
        xy = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

class MultipleDimensionInputModel(torch.nn.Module):
    def __init__(self):
        super(MultipleDimensionInputModel, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=8, out_features=6, bias=True)
        self.linear2 = torch.nn.Linear(in_features=6, out_features=4, bias=True)
        self.linear3 = torch.nn.Linear(in_features=4, out_features=1, bias=True)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x

def main():
    dataset = DiabetesDataset('/home/lsouba/dataset/diabetes/diabetes.csv.gz')
    trainLoader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    model = MultipleDimensionInputModel()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1000):
        running_loss = 0.0
        running_correct = 0.0
        total_samples = 0

        for inputs, labels in trainLoader:
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size
            with torch.no_grad():
                running_correct += ((torch.sigmoid(y_pred) >= 0.5) == labels).float().sum().item()
            total_samples += batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / total_samples
        epoch_acc = running_correct / total_samples
        if epoch % 100 == 0 or epoch == 999:
            print(f'Epoch {epoch} | loss = {epoch_loss:.8f} | acc = {epoch_acc:.3f}')

    with torch.no_grad():
        y_pred = model(dataset.x_data)
        acc = ((torch.sigmoid(y_pred) >= 0.5) == dataset.y_data).float().mean().item()
        print(f'Test acc = {acc:.3f}')

if __name__ == '__main__':
    main()
