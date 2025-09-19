import torch

from src.models import MultiScaleSRNet


def test_multi_scale_outputs_have_expected_resolution():
    model = MultiScaleSRNet()
    input_tensor = torch.rand(1, 3, 24, 24)
    outputs = model(input_tensor, return_all=True)
    assert set(outputs.keys()) == {2, 3, 4}
    assert outputs[2].shape[2:] == (48, 48)
    assert outputs[3].shape[2:] == (72, 72)
    assert outputs[4].shape[2:] == (96, 96)


def test_forward_specific_scale():
    model = MultiScaleSRNet()
    input_tensor = torch.rand(1, 3, 16, 16)
    out = model(input_tensor, scale=2)
    assert out.shape == (1, 3, 32, 32)
    assert (0.0 <= out).all() and (out <= 1.0).all()
