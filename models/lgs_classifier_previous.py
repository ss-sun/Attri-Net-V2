import torch.nn as nn
import torch


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes, downsample_ratio, type="previous"):
        super(LogisticRegressionModel, self).__init__()
        self.input_size = input_size
        self.downsample_ratio = downsample_ratio
        self.type = type

        if self.type != 'sum(abs(mx))':
            self.down = nn.AvgPool2d(kernel_size=self.downsample_ratio, stride=self.downsample_ratio)
            dim = int(self.input_size * self.input_size / (self.downsample_ratio * self.downsample_ratio))
            self.linear = nn.Linear(dim, num_classes)
        if self.type == 'sum(abs(mx))':
            self.linear = nn.Linear(1, num_classes)



    def forward(self, x):
        if self.type != 'sum(abs(mx))':
            x = self.down(x)
            x = torch.flatten(x, start_dim=1)
            out = self.linear(x)
        else:
            out = self.linear(x)
        return out
