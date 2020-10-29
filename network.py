import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self):
        super(DnCNN, self).__init__()
        # num_layers = 17
        num_layers = 3
        input_channels = output_channels = 1
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=input_channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=output_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out
