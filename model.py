import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

# TODO: Test later if aditionally to the offsets the network could benefit from learninf a mask too.

DEFAULT_WIDTHS = (16, 32, 64, 128, 256, 512)


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
    def __init__(self, in_channels=1, widths=DEFAULT_WIDTHS, kernel_size=3):
        super(Encoder, self).__init__()
        channels = [in_channels, *widths]
        self.stages = nn.ModuleList([
            DoubleConv(channels[i], channels[i + 1], kernel_size)
            for i in range(len(widths))
        ])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_connections = []
        for stage in self.stages:
            x = stage(x)
            skip_connections.append(x)
            x = self.pool(x)
        return skip_connections, x


class EdgePrior(nn.Module):
    """Learned edge map from the raw input, projected onto the bottleneck token grid."""

    def __init__(self, in_channels=1, embed_dim=512, kernel_size=3):
        super().__init__()
        self.edge_conv = nn.Conv2d(in_channels, 1, kernel_size, padding=kernel_size // 2)
        self.project = nn.Conv2d(1, embed_dim, kernel_size=1)

    def forward(self, x, out_hw):
        edge = torch.sigmoid(self.edge_conv(x))
        edge = nn.functional.interpolate(edge, size=out_hw, mode="bilinear", align_corners=False)
        edge = self.project(edge)
        return edge.flatten(2).transpose(1, 2)


class EGMHSA(nn.Module):
    """Multi-head self-attention with the edge prior added to Q and K in place of
    positional encodings."""

    def __init__(self, dim=512, heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, z, edge):
        b, c, h, w = z.shape
        tokens = z.flatten(2).transpose(1, 2)
        attended, _ = self.attention(tokens + edge, tokens + edge, tokens)
        tokens = self.norm(tokens + self.dropout(attended))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class Bottleneck(nn.Module):
    def __init__(self, channels=512, heads=4, dropout=0.1):
        super().__init__()
        self.conv = DoubleConv(channels, channels)
        self.attention1 = EGMHSA(channels, heads, dropout)
        self.attention2 = EGMHSA(channels, heads, dropout)

    def forward(self, x, edge):
        z = self.conv(x)
        z = self.attention1(z, edge)
        return self.attention2(z, edge)


class Decoder(nn.Module):
    def __init__(self, widths=DEFAULT_WIDTHS, bottleneck_channels=512, kernel_size=3):
        super().__init__()
        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()
        in_channels = bottleneck_channels
        for width in reversed(widths):
            self.ups.append(nn.ConvTranspose2d(in_channels, width, kernel_size=2, stride=2))
            self.convs.append(DoubleConv(2 * width, width, kernel_size))
            in_channels = width

    def forward(self, z, skips):
        for up, conv, skip in zip(self.ups, self.convs, reversed(skips)):
            z = up(z)
            z = conv(torch.cat([skip, z], dim=1))
        return z


class FilaNet(nn.Module):
    def __init__(self):
        super(FilaNet, self).__init__()
        # Define your model architecture here
        pass
