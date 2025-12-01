import torch.nn as nn

class DownConvLayer(nn.Module):
    def __init__(self, dim):
        super(DownConvLayer, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)