import torch
# import torch.nn.functional as F

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(in_features=1, out_features=1, bias=True)

    def forward(self, x):
        # y_pred = F.sigmoid(self.linear(x))
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

def main():
    x1_data = torch.Tensor([[1.0], [2.0], [3.0]])
    y1_data = torch.Tensor([[0.0], [0.0], [1.0]])
    x2_data = torch.Tensor([[1.0], [2.0], [5.0], [6.0]])
    y2_data = torch.Tensor([[0.0], [0.0], [1.0], [1.0]])
    x3_data = torch.Tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]])
    y3_data = torch.Tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]])

    epochNum = 1000
    criterion = torch.nn.BCELoss(reduction='sum')

    ##################################### x1,y1 #######################################
    model1 = LogisticRegressionModel()
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)

    for epoch in range(epochNum):
        y_pred1 = model1(x1_data)
        loss1 = criterion(y_pred1, y1_data)
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

    print(f'w = {model1.linear.weight.item():.3f}, b = {model1.linear.bias.item():.3f}')
    print(f'Predict (after training): f(10) = {model1(torch.Tensor([[10.0]])).item():.3f}')

    ##################################### x2,y2 #######################################
    model2 = LogisticRegressionModel()
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)

    for epoch in range(epochNum):
        y_pred2 = model2(x2_data)
        loss2 = criterion(y_pred2, y2_data)
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

    print(f'w = {model2.linear.weight.item():.3f}, b = {model2.linear.bias.item():.3f}')
    print(f'Predict (after training): f(10) = {model2(torch.Tensor([[10.0]])).item():.3f}')

    ##################################### x3,y3 #######################################
    model3 = LogisticRegressionModel()
    optimizer3 = torch.optim.SGD(model3.parameters(), lr=0.01)

    for epoch in range(epochNum):
        y_pred3 = model3(x3_data)
        loss3 = criterion(y_pred3, y3_data)
        optimizer3.zero_grad()
        loss3.backward()
        optimizer3.step()

    print(f'w = {model3.linear.weight.item():.3f}, b = {model3.linear.bias.item():.3f}')
    print(f'Predict (after training): f(10) = {model3(torch.Tensor([[10.0]])).item():.3f}')


    #####
    import numpy as np
    import matplotlib.pyplot as plt

    x = torch.linspace(0, 10, 200)
    x = torch.Tensor(x).view(200, 1)
    y1 = model1(x).data.numpy()
    y2 = model2(x).data.numpy()
    y3 = model3(x).data.numpy()
    plt.plot(x, y1, label='model1', color='blue')
    plt.plot(x, y2, label='model2', color='red')
    plt.plot(x, y3, label='model3', color='green')
    plt.plot([0, 10], [0.5, 0.5], c='r')
    plt.xlabel('Hours')
    plt.ylabel('Probability of Pass')
    plt.title('Logistic Regression Model')
    plt.legend()
    plt.grid(True)
    plt.savefig('logistic_regression_model.png', dpi=150)
    print('已保存: logistic_regression_model.png')


if __name__ == '__main__':
    main()
