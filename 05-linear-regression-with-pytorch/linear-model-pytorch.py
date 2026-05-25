import torch
import matplotlib.pyplot as plt

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(in_features=1, out_features=1, bias=True)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

def main():
    x_data = torch.Tensor([[1.0], [2.0], [3.0]])
    y_data = torch.Tensor([[2.0], [4.0], [6.0]])

    epochNum = 100

    # criterion = torch.nn.MSELoss(size_average=False) # deprecated
    criterion = torch.nn.MSELoss(reduction='sum')

    ##################################### SGD #######################################
    modelSGD = LinearModel()
    optimizerSGD = torch.optim.SGD(modelSGD.parameters(), lr=0.01)

    for epoch in range(epochNum):
        y_predSGD = modelSGD(x_data)
        lossSGD = criterion(y_predSGD, y_data)

        # if epoch % 10 == 0:
        #     print(f'Epoch {epoch} | loss = {loss.item():.8f}')

        optimizerSGD.zero_grad()
        lossSGD.backward()
        optimizerSGD.step()

    print(f'w = {modelSGD.linear.weight.item():.3f}, b = {modelSGD.linear.bias.item():.3f}')
    print(f'SGD: Predict (after training): f(4) = {modelSGD(torch.Tensor([[4.0]])).item():.3f}')

    ##################################### Adagrad #######################################
    modelAdagrad = LinearModel()
    optimizerAdagrad = torch.optim.Adagrad(modelAdagrad.parameters(), lr=0.01)
    for epoch in range(epochNum):
        y_predAdagrad = modelAdagrad(x_data)
        lossAdagrad = criterion(y_predAdagrad, y_data)

        optimizerAdagrad.zero_grad()
        lossAdagrad.backward()
        optimizerAdagrad.step()

    print(f'Adagrad: w = {modelAdagrad.linear.weight.item():.3f}, b = {modelAdagrad.linear.bias.item():.3f}')
    print(f'Adagrad: Predict (after training): f(4) = {modelAdagrad(torch.Tensor([[4.0]])).item():.3f}')

    ##################################### Adam #######################################
    modelAdam = LinearModel()
    optimizerAdam = torch.optim.Adam(modelAdam.parameters(), lr=0.01)
    for epoch in range(epochNum):
        y_predAdam = modelAdam(x_data)
        lossAdam = criterion(y_predAdam, y_data)

        optimizerAdam.zero_grad()
        lossAdam.backward()
        optimizerAdam.step()

    print(f'Adam: w = {modelAdam.linear.weight.item():.3f}, b = {modelAdam.linear.bias.item():.3f}')
    print(f'Adam: Predict (after training): f(4) = {modelAdam(torch.Tensor([[4.0]])).item():.3f}')

    ##################################### Adamax #######################################
    modelAdamax = LinearModel()
    optimizerAdamax = torch.optim.Adamax(modelAdamax.parameters(), lr=0.01)
    for epoch in range(epochNum):
        y_predAdamax = modelAdamax(x_data)
        lossAdamax = criterion(y_predAdamax, y_data)

        optimizerAdamax.zero_grad()
        lossAdamax.backward()
        optimizerAdamax.step()

    print(f'Adamax: w = {modelAdamax.linear.weight.item():.3f}, b = {modelAdamax.linear.bias.item():.3f}')
    print(f'Adamax: Predict (after training): f(4) = {modelAdamax(torch.Tensor([[4.0]])).item():.3f}')

    ##################################### ASGD #######################################
    modelASGD = LinearModel()
    optimizerASGD = torch.optim.ASGD(modelASGD.parameters(), lr=0.01)
    for epoch in range(epochNum):
        y_predASGD = modelASGD(x_data)
        lossASGD = criterion(y_predASGD, y_data)

        optimizerASGD.zero_grad()
        lossASGD.backward()
        optimizerASGD.step()

    print(f'ASGD: w = {modelASGD.linear.weight.item():.3f}, b = {modelASGD.linear.bias.item():.3f}')
    print(f'ASGD: Predict (after training): f(4) = {modelASGD(torch.Tensor([[4.0]])).item():.3f}')

    ##################################### LBFGS #######################################
    modelLBFGS = LinearModel()
    optimizerLBFGS = torch.optim.LBFGS(modelLBFGS.parameters(), lr=0.01)
    for epoch in range(epochNum):
        def closure():
            optimizerLBFGS.zero_grad()
            y_predLBFGS = modelLBFGS(x_data)
            lossLBFGS = criterion(y_predLBFGS, y_data)
            lossLBFGS.backward()
            return lossLBFGS
        optimizerLBFGS.step(closure)

    print(f'LBFGS: w = {modelLBFGS.linear.weight.item():.3f}, b = {modelLBFGS.linear.bias.item():.3f}')
    print(f'LBFGS: Predict (after training): f(4) = {modelLBFGS(torch.Tensor([[4.0]])).item():.3f}')

    ##################################### RMSprop #######################################
    modelRMSprop = LinearModel()
    optimizerRMSprop = torch.optim.RMSprop(modelRMSprop.parameters(), lr=0.01)
    for epoch in range(epochNum):
        y_predRMSprop = modelRMSprop(x_data)
        lossRMSprop = criterion(y_predRMSprop, y_data)

        optimizerRMSprop.zero_grad()
        lossRMSprop.backward()
        optimizerRMSprop.step()

    print(f'RMSprop: w = {modelRMSprop.linear.weight.item():.3f}, b = {modelRMSprop.linear.bias.item():.3f}')
    print(f'RMSprop: Predict (after training): f(4) = {modelRMSprop(torch.Tensor([[4.0]])).item():.3f}')

    ##################################### Rprop #######################################
    modelRprop = LinearModel()
    optimizerRprop = torch.optim.Rprop(modelRprop.parameters(), lr=0.01)
    for epoch in range(epochNum):
        y_predRprop = modelRprop(x_data)
        lossRprop = criterion(y_predRprop, y_data)

        optimizerRprop.zero_grad()
        lossRprop.backward()
        optimizerRprop.step()

    print(f'Rprop: w = {modelRprop.linear.weight.item():.3f}, b = {modelRprop.linear.bias.item():.3f}')
    print(f'Rprop: Predict (after training): f(4) = {modelRprop(torch.Tensor([[4.0]])).item():.3f}')

    # 收集各优化器的 loss 曲线
    def train_and_collect_losses(model_cls, optimizer_cls, optimizer_name, x_data, y_data, epochs=epochNum, lr=0.01):
        model = model_cls()
        optimizer = optimizer_cls(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        losses = []
        for epoch in range(epochs):
            y_pred_optimizer = model(x_data)
            loss_optimizer = criterion(y_pred_optimizer, y_data)
            optimizer.zero_grad()
            loss_optimizer.backward()
            optimizer.step()
            losses.append(loss_optimizer.item())
        return losses

    # 各优化器的 loss 曲线
    losses_dict = {}

    # SGD
    losses_dict['SGD'] = train_and_collect_losses(LinearModel, torch.optim.SGD, "SGD", x_data, y_data)
    # Momentum
    losses_dict['Momentum'] = train_and_collect_losses(LinearModel, lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9), "Momentum", x_data, y_data)
    # Adam
    losses_dict['Adam'] = train_and_collect_losses(LinearModel, torch.optim.Adam, "Adam", x_data, y_data)
    # Adagrad
    losses_dict['Adagrad'] = train_and_collect_losses(LinearModel, torch.optim.Adagrad, "Adagrad", x_data, y_data)
    # Adadelta
    losses_dict['Adadelta'] = train_and_collect_losses(LinearModel, torch.optim.Adadelta, "Adadelta", x_data, y_data)
    # ASGD
    losses_dict['ASGD'] = train_and_collect_losses(LinearModel, torch.optim.ASGD, "ASGD", x_data, y_data)
    # LBFGS
    # LBFGS 需要 closure
    def lbfgs_losses(epochs=epochNum):
        model = LinearModel()
        optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        losses = []
        for epoch in range(epochs):
            def closure():
                optimizer.zero_grad()
                output_lbfgs = model(x_data)
                loss_lbfgs = criterion(output_lbfgs, y_data)
                loss_lbfgs.backward()
                return loss_lbfgs
            loss_lbfgs = optimizer.step(closure)
            losses.append(loss_lbfgs.item())
        return losses
    losses_dict['LBFGS'] = lbfgs_losses(epochs=epochNum)
    # RMSprop
    losses_dict['RMSprop'] = train_and_collect_losses(LinearModel, torch.optim.RMSprop, "RMSprop", x_data, y_data)
    # Rprop
    # Rprop requires float dtype and doesn't support vectorized learning as well as others, but for demo:
    losses_dict['Rprop'] = train_and_collect_losses(LinearModel, torch.optim.Rprop, "Rprop", x_data, y_data)

    # 画 loss 曲线
    plt.figure(figsize=(12,8))
    for name, losses in losses_dict.items():
        plt.plot(range(1, len(losses)+1), losses, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves for Different Optimizers')
    plt.legend()
    plt.grid(True)
    plt.savefig('all_optim_loss_curves.png', dpi=150)
    print('已保存: all_optim_loss_curves.png')

if __name__ == '__main__':
    main()
