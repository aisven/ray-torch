import torch
import torch.nn.functional as tf

from ray_torch.camera.camera import eye
from ray_torch.camera.camera import img_grid
from ray_torch.camera.camera import middle_pixel_index
from ray_torch.camera.camera import n_pixels
from ray_torch.camera.camera import px_ll
from ray_torch.camera.camera import px_lr
from ray_torch.camera.camera import px_ul
from ray_torch.camera.camera import px_ur
from ray_torch.camera.camera import resx_int_py
from ray_torch.camera.camera import resy
from ray_torch.camera.camera import resy_int_py
from ray_torch.constant.constant import one_dot_zero
from ray_torch.constant.constant import one_over_255
from ray_torch.constant.constant import two_55
from ray_torch.constant.constant import zero_dot_zero
from ray_torch.constant.constant import zero_vector_float
from ray_torch.constant.constant import zero_vector_int
from ray_torch.intersection.intersection import intersect_rays_with_spheres
from ray_torch.plot.plot import plot_rgb_image
from ray_torch.plot.plot import plot_vectors_with_color_by_norm
from ray_torch.plot.plot import plot_vectors_with_color_by_z_value
from ray_torch.utility.utility import create_tensor_on_device
from ray_torch.utility.utility import device
from ray_torch.utility.utility import is_float_tensor_on_device
from ray_torch.utility.utility import see
from ray_torch.utility.utility import see_more


# global PyTorch settings

# disable gradient tracking not needed in this ray tracer to save compute
torch.set_grad_enabled(False)

# global PyVista settings

# render on client-side instead of Jupyter server
# pv.set_jupyter_backend('client')

# check for Metal Performance Shaders in case of macOS just out of curiosity
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")


# define point light(s)


def create_point_lights_1():
    point_lights_position_py = [[-10.0, 10.0, 0.0]]
    point_lights_position_pt = torch.tensor(point_lights_position_py, dtype=torch.float, requires_grad=False)
    n_point_lights = point_lights_position_pt.shape[0]
    point_lights_rgb_py = [[220, 190, 120]]
    point_lights_rgb_pt = torch.tensor(point_lights_rgb_py, dtype=torch.int, requires_grad=False)
    assert point_lights_position_pt.shape == (n_point_lights, 3)
    assert point_lights_rgb_pt.shape == (n_point_lights, 3)
    return n_point_lights, point_lights_position_pt, point_lights_rgb_pt    


n_point_lights, point_lights_position, point_lights_rgb = create_point_lights_1()
point_lights_position = point_lights_position.to(device)
point_lights_rgb = point_lights_rgb.to(device)

see("point_lights_position", point_lights_position)
assert is_float_tensor_on_device(point_lights_position)

see("point_lights_rgb", point_lights_rgb)


# define spheres
# each sphere is defined by its center in camera system coordinates and its radius


def create_spheres_1():
    spheres_center_py = [[-1.0, 0.0, 8.0], [2.0, -2.0, 12.0], [0.0, 0.0, 20.0], [-8.0, 0.0, 10.0]]
    spheres_center_pt = torch.tensor(spheres_center_py, dtype=torch.float, requires_grad=False)
    n_spheres = spheres_center_pt.shape[0]
    spheres_radius_py = [1.0, 5.0, 10.0, 2.0]
    spheres_radius_pt = torch.tensor(spheres_radius_py, dtype=torch.float, requires_grad=False)
    spheres_rgb_py = [[150, 90, 200], [255, 144, 0], [255, 255, 255], [255, 0, 0]]
    spheres_rgb_pt = torch.tensor(spheres_rgb_py, dtype=torch.int, requires_grad=False)
    assert spheres_center_pt.shape == (n_spheres, 3)
    assert spheres_radius_pt.shape == (n_spheres,)
    assert spheres_rgb_pt.shape == (n_spheres, 3)
    return n_spheres, spheres_center_pt, spheres_radius_pt, spheres_rgb_pt


n_spheres, spheres_center, spheres_radius, spheres_rgb = create_spheres_1()
spheres_center = spheres_center.to(device)
spheres_radius = spheres_radius.to(device)
spheres_rgb = spheres_rgb.to(device)

see("spheres_center", spheres_center)
assert is_float_tensor_on_device(spheres_center)

see("spheres_radius", spheres_radius)
assert is_float_tensor_on_device(spheres_radius)

see("spheres_rgb", spheres_rgb)


# compute primary rays

primary_ray_vectors = img_grid - eye
see("primary_ray_vectors", primary_ray_vectors)
assert is_float_tensor_on_device(primary_ray_vectors)

# sanity check on the edges of the image grid

primary_ray_vector_px_ul = primary_ray_vectors[0]
primary_ray_vector_px_ll = primary_ray_vectors[int(resy.item()) - 1]
primary_ray_vector_px_ur = primary_ray_vectors[primary_ray_vectors.shape[0] - int(resy.item())]
primary_ray_vector_px_lr = primary_ray_vectors[primary_ray_vectors.shape[0] - 1]

# compare vectors calculated the old and the modern way

assert torch.allclose(primary_ray_vector_px_ul, px_ul)
assert torch.allclose(primary_ray_vector_px_ul, px_ul)
assert torch.allclose(primary_ray_vector_px_ul, px_ul)
assert torch.allclose(primary_ray_vector_px_ul, px_ul)

print(f"primary_ray_vectors[middle_pixel_index]={primary_ray_vectors[middle_pixel_index]}")

# normalize the primary ray vectors so that they become unit vectors

primary_ray_vectors_unit = tf.normalize(primary_ray_vectors)
see("primary_ray_vectors_unit", primary_ray_vectors_unit)
assert is_float_tensor_on_device(primary_ray_vectors_unit)

# check that for each primary ray vector the Euclidean norm sqrt(x^2 + y^2 + z^2) is 1.0
assert torch.allclose(torch.norm(primary_ray_vectors_unit, dim=1), one_dot_zero)

# it follows that for each primary ray vector the squared norm x^2 + y^2 + z^2 is also 1.0
assert torch.allclose(torch.mul(primary_ray_vectors_unit, primary_ray_vectors_unit).sum(dim=1), one_dot_zero)

# perform sanity checks just to be sure

primary_ray_vector_px_ul_unit = primary_ray_vectors_unit[0]
primary_ray_vector_px_ll_unit = primary_ray_vectors_unit[int(resy.item()) - 1]
primary_ray_vector_px_ur_unit = primary_ray_vectors_unit[primary_ray_vectors.shape[0] - int(resy.item())]
primary_ray_vector_px_lr_unit = primary_ray_vectors_unit[primary_ray_vectors.shape[0] - 1]

# compare to vectors normalized individually

primary_ray_vector_px_ul_unit_alt = tf.normalize(primary_ray_vector_px_ul, dim=0)
primary_ray_vector_px_ll_unit_alt = tf.normalize(primary_ray_vector_px_ll, dim=0)
primary_ray_vector_px_ur_unit_alt = tf.normalize(primary_ray_vector_px_ur, dim=0)
primary_ray_vector_px_lr_unit_alt = tf.normalize(primary_ray_vector_px_lr, dim=0)

assert torch.allclose(primary_ray_vector_px_ul_unit, primary_ray_vector_px_ul_unit_alt)
assert torch.allclose(primary_ray_vector_px_ll_unit, primary_ray_vector_px_ll_unit_alt)
assert torch.allclose(primary_ray_vector_px_ur_unit, primary_ray_vector_px_ur_unit_alt)
assert torch.allclose(primary_ray_vector_px_lr_unit, primary_ray_vector_px_lr_unit_alt)

# compare to vectors calculated the old way

px_ul_unit = tf.normalize(px_ul, dim=0)
px_ll_unit = tf.normalize(px_ll, dim=0)
px_ur_unit = tf.normalize(px_ur, dim=0)
px_lr_unit = tf.normalize(px_lr, dim=0)

assert torch.allclose(primary_ray_vector_px_ul_unit, px_ul_unit)
assert torch.allclose(primary_ray_vector_px_ll_unit, px_ll_unit)
assert torch.allclose(primary_ray_vector_px_ur_unit, px_ur_unit)
assert torch.allclose(primary_ray_vector_px_lr_unit, px_lr_unit)

print(f"primary_ray_vectors_unit[middle_pixel_index]={primary_ray_vectors_unit[middle_pixel_index]}")

# define function to compute intersections of rays with spheres

# make a tensor that contains one copy of eye per sphere
eyes_spheres_center = eye.unsqueeze(0).repeat(n_spheres, 1)
see("eyes_spheres_center", eyes_spheres_center, False)
assert is_float_tensor_on_device(eyes_spheres_center)
assert eyes_spheres_center.shape == (n_spheres, 3)

# make a tensor that contains one copy of eye per primary ray
eyes_pixels = eye.unsqueeze(0).repeat(n_pixels, 1)
assert is_float_tensor_on_device(eyes_pixels)
assert eyes_pixels.shape == (n_pixels, 3)
see("eyes_pixels", eyes_pixels, False)

points_hit, surface_normals_hit, surface_normals_hit_unit, spheres_index_hit, spheres_center_hit, background_mask, foreground_mask, foreground_mask_with_0 = intersect_rays_with_spheres(n_pixels, n_spheres, eyes_spheres_center, spheres_center, spheres_radius, primary_ray_vectors_unit)

print(f"spheres_index_hit[middle_pixel_index]={spheres_index_hit[middle_pixel_index]}")

print(f"spheres_center_hit[middle_pixel_index]={spheres_center_hit[middle_pixel_index]}")

print(f"points_hit[middle_pixel_index]={points_hit[middle_pixel_index]}")

print(f"surface_normals_hit[middle_pixel_index]={surface_normals_hit[middle_pixel_index]}")

print(f"surface_normals_hit_unit[middle_pixel_index]={surface_normals_hit_unit[middle_pixel_index]}")

spheres_rgb_hit = spheres_rgb[spheres_index_hit]
see("spheres_rgb_hit", spheres_rgb_hit)
assert spheres_rgb_hit.shape == (n_pixels, 3)


# compute illumination based on point lights, intersections, surface normals


def compute_lighting_diffuse_component_1_point_light(points_hit, foreground_mask, background_mask, surface_normals_hit_unit, spheres_rgb, spheres_index_hit, point_light_position, point_light_rgb):
    spheres_rgb_hit = spheres_rgb[spheres_index_hit]
    spheres_rgb_hit_01 = torch.mul(spheres_rgb_hit, one_over_255)
    point_light_rgb_01 = torch.mul(point_light_rgb, one_over_255)
    # weighting factor for diffuse component
    k_d = create_tensor_on_device(0.7)
    # number of intersections
    n_points_hit = points_hit.shape[0]
    # make a tensor that contains one copy of the point light position per intersection
    point_light_position_per_point_hit = point_light_position.unsqueeze(0).repeat(n_points_hit, 1)
    assert is_float_tensor_on_device(point_light_position_per_point_hit)
    assert point_light_position_per_point_hit.shape == (n_points_hit, 3)
    see("point_light_position_per_point_hit", point_light_position_per_point_hit, False)    
    # compute the direction vectors from intersections to point light position
    point_light_rays = point_light_position_per_point_hit - points_hit
    point_light_rays[background_mask] = zero_dot_zero
    see_more("point_light_rays", point_light_rays, True)
    point_light_rays_unit = tf.normalize(point_light_rays)
    point_light_rays_unit[background_mask] = zero_dot_zero
    see_more("point_light_rays_unit", point_light_rays_unit, True)
    # sanity check
    point_light_rays_unit_norm = torch.norm(point_light_rays_unit, p=2, dim=1, keepdim=True)
    see_more("point_light_rays_unit_norm", point_light_rays_unit_norm, False)
    assert is_float_tensor_on_device(point_light_rays_unit_norm)
    assert torch.allclose(point_light_rays_unit_norm[foreground_mask], one_dot_zero)
    assert torch.allclose(point_light_rays_unit_norm[background_mask], zero_vector_float)

    see_more("surface_normals_hit_unit", surface_normals_hit_unit, True)
    see_more("point_light_rays", point_light_rays, True)

    l_dot_n = torch.sum(torch.mul(surface_normals_hit_unit, point_light_rays_unit), dim=1)
    assert l_dot_n.shape == (n_points_hit,)
    see_more("l_dot_n", l_dot_n, True)

    l_dot_n_clamped = torch.maximum(l_dot_n, zero_dot_zero)
    assert l_dot_n_clamped.shape == (n_points_hit,)
    see_more("l_dot_n_clamped", l_dot_n_clamped, True)

    colors_d = torch.mul(torch.mul(l_dot_n_clamped, point_light_rgb_01.unsqueeze(1)).t(), one_dot_zero)
    see_more("colors_d", colors_d, True)
    assert colors_d.shape == (n_points_hit, 3)

    colors_d_weighted_01 = torch.mul(colors_d, k_d)
    see_more("colors_d_weighted_01", colors_d_weighted_01, True)
    assert colors_d_weighted_01.shape == (n_points_hit, 3)

    assert colors_d_weighted_01.shape == spheres_rgb_hit_01.shape
    colors_d_mixed_01 = torch.mul(colors_d_weighted_01, spheres_rgb_hit_01)

    colors_d_mixed = torch.mul(colors_d_mixed_01, two_55).to(torch.int)
    assert colors_d_mixed.shape == spheres_rgb_hit_01.shape

    return point_light_rays, point_light_rays_unit, colors_d_mixed_01, colors_d_mixed


def compute_lighting_specular_component_1_point_light(points_hit, surface_normals_hit_unit, eye, spheres_rgb, point_light_position, point_light_rgb):
    # weighting factor for specular component
    k_s = 0.3


point_light_rays, point_light_rays_unit, colors_d_mixed_01, colors_d_mixed = compute_lighting_diffuse_component_1_point_light(points_hit, foreground_mask, background_mask, surface_normals_hit_unit, spheres_rgb, spheres_index_hit, point_lights_position[0], point_lights_rgb[0])

print(f"colors_d_mixed_01[middle_pixel_index]={colors_d_mixed_01[middle_pixel_index]}")

# plots

plot_rgb_image(spheres_rgb_hit, background_mask, resx_int_py, resy_int_py)
plot_rgb_image(colors_d_mixed, background_mask, resx_int_py, resy_int_py)

plot_all = False

if plot_all:
    plot_vectors_with_color_by_z_value(points_hit, foreground_mask_with_0, [6, 16], 'terrain')
    plot_vectors_with_color_by_norm(surface_normals_hit, foreground_mask_with_0, [0, 12], 'terrain')
    plot_vectors_with_color_by_norm(point_light_rays, foreground_mask_with_0, [12, 30], 'terrain')
