import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

# TODO: Test later if aditionally to the offsets the network could benefit from learninf a mask too.

class FilaNet(nn.Module):
    def __init__(self):
        super(FilaNet, self).__init__()
        # Define your model architecture here
        pass

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
        
class EncoderStage(nn.Module):
    """Two stride-1 DeformedConvolution blocks: in_channels -> out_channels -> out_channels."""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(EncoderStage, self).__init__()
        self.conv1 = DeformedConvolution(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.conv2 = DeformedConvolution(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=1, stage_channels=(64, 128, 256, 512), kernel_size=3):
        super(Encoder, self).__init__()
        channels = [in_channels, *stage_channels]
        self.stages = nn.ModuleList([
            EncoderStage(channels[i], channels[i + 1], kernel_size)
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

class MHSA(nn.Module):
    def __init__(self):
        super(MHSA, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1, batch_first=True)
    
    def forward(self, x, edges):
        q = x + edges
        k = x + edges
        v = x
        out = self.attention(q, k, v)
        return out  

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Bottleneck, self).__init__()
        self.stage = EncoderStage(in_channels, out_channels, kernel_size)
        self.edge_tensor = EdgePrior(out_channels, 32, 32)  # Example dimensions, adjust as needed
        self.mhsa = MHSA()
    
    def forward(self, x, edges):
        x = self.stage(x)
        edges = self.edge_tensor.forward(x)
        x = self.mhsa(x, edges)
        return x

class EdgePrior(nn.Module):
    def __init__(self, bottleneck_channels, bottleneck_height, bottleneck_width):
        super(EdgePrior, self).__init__()
        self.bottleneck_size = (bottleneck_height, bottleneck_width)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.sigm = nn.Sigmoid()
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=bottleneck_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.sigm(x)
        x = nn.functional.interpolate(x, size=self.bottleneck_size, mode='bilinear', align_corners=False)
        out = self.conv2(x)
        return out

class DecoderStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        # Upsample by two
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = EncoderStage(out_channels, out_channels)
    
    def forward(self, x, connection):
        x = self.upsample(x)
        x = torch.cat([x, connection], dim=1)
        x = self.conv_block(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels = 512, stage_channels = (256, 128, 64, 1)):
        super(Decoder, self).__init__()
        channels = [in_channels, *stage_channels]
        self.stages = nn.ModuleList([
            DecoderStage(stage_channels[i], channels[i + 1])
            for i in range(len(stage_channels) - 1)
        ])