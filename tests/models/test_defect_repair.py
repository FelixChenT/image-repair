import torch

from src.models import DefectRepairModule


def test_defect_repair_module_outputs_mask_and_image():
    model = DefectRepairModule()
    input_tensor = torch.rand(2, 3, 64, 64)
    repaired, mask = model(input_tensor)
    assert repaired.shape == input_tensor.shape
    assert mask.shape == (2, 1, 64, 64)
    assert (0.0 <= repaired).all() and (repaired <= 1.0).all()
    assert (0.0 <= mask).all() and (mask <= 1.0).all()
