import torch

from ray_torch.constant.constant import device
from ray_torch.constant.constant import one_over_255


def create_point_lights_1():
    point_lights_position_py = [[-10.0, 8.0, -2.0]]
    point_lights_position_pt = torch.tensor(
        point_lights_position_py, dtype=torch.float, requires_grad=False, device=device
    )
    n_point_lights = point_lights_position_pt.shape[0]
    point_lights_rgb_py = [[220, 190, 120]]
    point_lights_rgb_pt = torch.tensor(point_lights_rgb_py, dtype=torch.int, requires_grad=False, device=device)
    assert point_lights_position_pt.shape == (n_point_lights, 3)
    assert point_lights_rgb_pt.shape == (n_point_lights, 3)
    point_lights_rgb_01_pt = torch.mul(point_lights_rgb_pt, one_over_255)
    return n_point_lights, point_lights_position_pt, point_lights_rgb_pt, point_lights_rgb_01_pt
