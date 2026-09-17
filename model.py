import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

# TODO: Test later if aditionally to the offsets the network could benefit from learninf a mask too.


class DeformedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        """Deformed Convolution layers recieve the output of a regular convolution
        and use it as a learned offset map that is then aplied in the deform_conv2d
        """
        super(DeformedConvolution, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # Keeps spatial dimensions the same and ensure out_channels have the necessary size (ex. for a kernel of 3 the out channels would be 18)
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, padding=kernel_size//2)
        # Stride 1: keeps spatial dimensions the same, changes channel count. Downsampling is a separate pooling step.
        self.deform_conv = DeformConv2d(in_channels, self.out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.batch_norm = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        offsets = self.offset_conv(x)
        x = self.deform_conv(x, offsets)
        x = self.batch_norm(x)
        return self.relu(x)


class DoubleConv(nn.Module):
    """Two stride-1 deformable convs: in_channels -> out_channels -> out_channels."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = DeformedConvolution(in_channels, out_channels, kernel_size)
        self.conv2 = DeformedConvolution(out_channels, out_channels, kernel_size)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class Encoder(nn.Module):
    def __init__(self, in_channels=1, stage_channels=(64, 128, 256, 512), kernel_size=3):
        super(Encoder, self).__init__()
        channels = [in_channels, *stage_channels]
        self.stages = nn.ModuleList([
            DoubleConv(channels[i], channels[i + 1], kernel_size)
            for i in range(len(stage_channels))
        ])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_connections = []
        for stage in self.stages:
            x = stage(x)
            skip_connections.append(x)
            x = self.pool(x)
        return skip_connections, x


class FilaNet(nn.Module):
    def __init__(self):
        super(FilaNet, self).__init__()
        # Define your model architecture here
        pass
