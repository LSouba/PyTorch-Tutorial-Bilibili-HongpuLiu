import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionModule(nn.Module):
    def __init__(self, in_channels):
        super(InceptionModule, self).__init__()
        self.branch10 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1)   # 1x1 convolution

        self.branch20 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1)   # 5x5 convolution
        self.branch21 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=5, padding=2)

        self.branch30 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1)   # 3x3 convolution
        self.branch31 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, padding=1)
        self.branch32 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, padding=1)

        self.branch40 = F.avg_pool2d(in_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.branch41 = nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=1)   # 1x1 convolution

    def forward(self, x):
        output1 = self.branch10(x)

        output2 = self.branch20(x)
        output2 = self.branch21(output2)

        output3 = self.branch30(x)
        output3 = self.branch31(output3)
        output3 = self.branch32(output3)

        output4 = self.branch40(x)
        output4 = self.branch41(output4)

        return torch.cat([output1, output2, output3, output4], dim=1)

class AdvancedCNN(nn.Module):
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=88, out_channels=20, kernel_size=5)

        self.inception1 = InceptionModule(in_channels=10)
        self.inception2 = InceptionModule(in_channels=20)

        self.pool   = nn.MaxPool2d(2)
        self.linear = nn.Linear(in_features=1408, out_features=10)

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.pool(self.conv1(x)))  # [B, 1, 28, 28]  -> [B, 10, 12, 12]
        x = self.inception1(x)                # [B, 10, 12, 12] -> [B, 88, 12, 12]
        x = F.relu(self.pool(self.conv2(x)))  # [B, 88, 12, 12] -> [B, 20, 4, 4]
        x = self.inception2(x)                # [B, 20, 4, 4]   -> [B, 88, 4, 4]
        x = x.view(batch_size, -1)            # [B, 88, 4, 4]   -> [B, 1408]
        x = self.linear(x)                    # [B, 1408]       -> [B, 10]
        return x
