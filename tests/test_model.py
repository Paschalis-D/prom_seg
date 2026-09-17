import torch
import torch.nn as nn

from model import DeformedConvolution


def test_deformed_convolution_preserves_spatial_dims():
    block = DeformedConvolution(in_channels=3, out_channels=8, kernel_size=3)
    y = block(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 8, 32, 32)


def test_deformed_convolution_handles_non_power_of_two():
    block = DeformedConvolution(in_channels=1, out_channels=4, kernel_size=3)
    y = block(torch.randn(2, 1, 30, 30))
    assert y.shape == (2, 4, 30, 30)


def test_deformed_convolution_offsets_match_kernel():
    block = DeformedConvolution(in_channels=3, out_channels=8, kernel_size=3)
    assert block.offset_conv.out_channels == 2 * 3 * 3
    assert block.deform_conv.stride == (1, 1)
