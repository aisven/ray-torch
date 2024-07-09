import torch

from ray_torch.constant.constant import device
from ray_torch.constant.constant import one_over_255


def create_spheres_1():
    # the geometry of each sphere is defined by its center in camera system coordinates and its radius
    spheres_center_py = [[-1.0, 0.0, 8.0], [2.0, -2.0, 12.0], [0.0, 0.0, 16.0], [-5.0, 0.0, 10.0], [2.2, -2.2, 6.7]]
    spheres_center_pt = torch.tensor(spheres_center_py, dtype=torch.float, requires_grad=False, device=device)
    n_spheres = spheres_center_pt.shape[0]
    spheres_radius_py = [1.0, 4.0, 6.0, 1.0, 1.0]
    spheres_radius_pt = torch.tensor(spheres_radius_py, dtype=torch.float, requires_grad=False, device=device)
    # colors given as integers representing RGB and equivalently in real values in range 0.0 to 1.0
    spheres_rgb_py = [[150, 90, 200], [255, 144, 0], [200, 155, 255], [255, 0, 0], [123, 132, 231]]
    spheres_rgb_pt = torch.tensor(spheres_rgb_py, dtype=torch.int, requires_grad=False, device=device)
    assert spheres_center_pt.shape == (n_spheres, 3)
    assert spheres_radius_pt.shape == (n_spheres,)
    assert spheres_rgb_pt.shape == (n_spheres, 3)
    spheres_rgb_01_pt = torch.mul(spheres_rgb_pt, one_over_255)
    return n_spheres, spheres_center_pt, spheres_radius_pt, spheres_rgb_pt, spheres_rgb_01_pt
