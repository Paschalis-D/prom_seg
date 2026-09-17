import torch
import torch.nn as nn

from model import DEFAULT_WIDTHS, DeformedConvolution, DoubleConv, EdgePrior, Encoder


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


def test_double_conv_changes_channels_once():
    block = DoubleConv(in_channels=4, out_channels=16)
    y = block(torch.randn(2, 4, 32, 32))
    assert y.shape == (2, 16, 32, 32)


def test_double_conv_second_layer_keeps_width():
    block = DoubleConv(in_channels=4, out_channels=16)
    assert block.conv1.deform_conv.in_channels == 4
    assert block.conv1.deform_conv.out_channels == 16
    assert block.conv2.deform_conv.in_channels == 16
    assert block.conv2.deform_conv.out_channels == 16


def test_encoder_returns_one_skip_per_stage():
    enc = Encoder()
    skips, out = enc(torch.randn(2, 1, 256, 256))
    assert len(skips) == 6
    expected = [
        (2, 16, 256, 256),
        (2, 32, 128, 128),
        (2, 64, 64, 64),
        (2, 128, 32, 32),
        (2, 256, 16, 16),
        (2, 512, 8, 8),
    ]
    assert [tuple(s.shape) for s in skips] == expected


def test_encoder_bottleneck_is_input_over_64():
    enc = Encoder()
    _, out = enc(torch.randn(2, 1, 256, 256))
    assert out.shape == (2, 512, 4, 4)


def test_encoder_has_one_stage_per_width():
    enc = Encoder()
    assert len(enc.stages) == len(DEFAULT_WIDTHS)
    assert DEFAULT_WIDTHS == (16, 32, 64, 128, 256, 512)


def test_edge_prior_produces_one_token_per_bottleneck_cell():
    ep = EdgePrior(in_channels=1, embed_dim=512)
    e = ep(torch.randn(2, 1, 256, 256), out_hw=(4, 4))
    assert e.shape == (2, 16, 512)


def test_edge_prior_is_resolution_agnostic():
    ep = EdgePrior(in_channels=1, embed_dim=64)
    assert ep(torch.randn(2, 1, 256, 256), out_hw=(8, 8)).shape == (2, 64, 64)
    assert ep(torch.randn(2, 1, 128, 128), out_hw=(2, 2)).shape == (2, 4, 64)


def test_edge_prior_squeezes_to_single_channel_before_projection():
    ep = EdgePrior(in_channels=1, embed_dim=512)
    assert ep.edge_conv.out_channels == 1
    assert ep.project.kernel_size == (1, 1)
    assert ep.project.out_channels == 512
