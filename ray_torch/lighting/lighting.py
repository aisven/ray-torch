import torch
import torch.nn.functional as tf

from ray_torch.constant.constant import minus_one_dot_zero
from ray_torch.constant.constant import one_dot_zero
from ray_torch.constant.constant import two_55
from ray_torch.constant.constant import two_dot_zero
from ray_torch.constant.constant import zero_dot_zero
from ray_torch.constant.constant import zero_vector_float
from ray_torch.utility.utility import create_tensor_on_device
from ray_torch.utility.utility import is_float_tensor_on_device
from ray_torch.utility.utility import see_more


def compute_lighting_diffuse_component_1_point_light(
    l_dot_n,
    points_hit,
    foreground_mask,
    background_mask,
    surface_normals_hit_unit,
    spheres_rgb_01,
    spheres_index_hit,
    point_light_position,
    point_light_rgb_01,
):
    spheres_rgb_hit_01 = spheres_rgb_01[spheres_index_hit]
    # weighting factor for diffuse component
    k_d = create_tensor_on_device(0.6)
    # number of intersections
    n_points_hit = points_hit.shape[0]

    l_dot_n_clamped = torch.maximum(l_dot_n, zero_dot_zero)
    assert is_float_tensor_on_device(l_dot_n_clamped)
    assert l_dot_n_clamped.shape == (n_points_hit,)
    see_more("l_dot_n_clamped", l_dot_n_clamped, True)

    colors_d = torch.mul(point_light_rgb_01.unsqueeze(1), l_dot_n_clamped).t()
    assert is_float_tensor_on_device(colors_d)
    see_more("colors_d", colors_d, True)
    assert colors_d.shape == (n_points_hit, 3)

    colors_d_weighted_01 = torch.mul(colors_d, k_d)
    assert is_float_tensor_on_device(colors_d_weighted_01)
    see_more("colors_d_weighted_01", colors_d_weighted_01, True)
    assert colors_d_weighted_01.shape == (n_points_hit, 3)

    assert colors_d_weighted_01.shape == spheres_rgb_hit_01.shape
    colors_d_mixed_01 = torch.mul(colors_d_weighted_01, spheres_rgb_hit_01)

    colors_d_mixed = torch.mul(colors_d_mixed_01, two_55).to(torch.int)
    assert colors_d_mixed.shape == spheres_rgb_hit_01.shape

    return colors_d_mixed_01, colors_d_mixed


def compute_lighting_specular_component_1_point_light(
    l_dot_n,
    point_light_rays_unit,
    points_hit,
    foreground_mask,
    background_mask,
    surface_normals_hit_unit,
    spheres_rgb_01,
    spheres_index_hit,
    point_light_position,
    point_light_rgb_01,
    primary_ray_vectors_unit,
):
    # spheres_rgb_hit_01 = spheres_rgb_01[spheres_index_hit]
    # weighting factor for specular component
    k_s = create_tensor_on_device(0.4)
    # number of intersections
    n_points_hit = points_hit.shape[0]
    # shininess
    shininess = create_tensor_on_device(100.0)

    two_l_dot_n = torch.mul(l_dot_n, two_dot_zero)
    assert is_float_tensor_on_device(two_l_dot_n)
    assert two_l_dot_n.shape == (n_points_hit,)
    see_more("two_l_dot_n", two_l_dot_n, True)

    see_more("surface_normals_hit_unit", surface_normals_hit_unit, True)

    surface_normals_hit_scaled = torch.mul(two_l_dot_n.unsqueeze(1), surface_normals_hit_unit)
    assert is_float_tensor_on_device(surface_normals_hit_scaled)
    assert surface_normals_hit_scaled.shape == (n_points_hit, 3)
    see_more("surface_normals_hit_scaled", surface_normals_hit_scaled, True)

    reflection_vectors = surface_normals_hit_scaled - point_light_rays_unit
    assert is_float_tensor_on_device(reflection_vectors)
    assert reflection_vectors.shape == (n_points_hit, 3)
    see_more("reflection_vectors", reflection_vectors, True)

    view_vectors = torch.mul(primary_ray_vectors_unit, minus_one_dot_zero)
    assert is_float_tensor_on_device(view_vectors)
    assert view_vectors.shape == (n_points_hit, 3)
    see_more("view_vectors", view_vectors, True)

    r_dot_v = torch.sum(torch.mul(reflection_vectors, view_vectors), dim=1)
    assert is_float_tensor_on_device(r_dot_v)
    assert r_dot_v.shape == (n_points_hit,)
    r_dot_v[background_mask] = zero_dot_zero
    r_dot_v[torch.isnan(r_dot_v)] = zero_dot_zero
    see_more("r_dot_v", r_dot_v, True)

    r_dot_v_clamped = torch.maximum(r_dot_v, zero_dot_zero)
    assert is_float_tensor_on_device(r_dot_v_clamped)
    assert r_dot_v_clamped.shape == (n_points_hit,)
    see_more("r_dot_v_clamped", r_dot_v_clamped, True)

    r_dot_v_shine = torch.pow(r_dot_v_clamped, shininess)
    assert is_float_tensor_on_device(r_dot_v_shine)
    assert r_dot_v_shine.shape == (n_points_hit,)
    see_more("r_dot_v_shine", r_dot_v_shine, True)

    colors_s = torch.mul(point_light_rgb_01.unsqueeze(1), r_dot_v_shine).t()
    assert is_float_tensor_on_device(colors_s)
    see_more("colors_s", colors_s, True)
    assert colors_s.shape == (n_points_hit, 3)

    colors_s_weighted_01 = torch.mul(colors_s, k_s)
    assert is_float_tensor_on_device(colors_s_weighted_01)
    see_more("colors_s_weighted_01", colors_s_weighted_01, True)
    assert colors_s_weighted_01.shape == (n_points_hit, 3)

    colors_s_weighted = torch.mul(colors_s_weighted_01, two_55).to(torch.int)

    return colors_s_weighted_01, colors_s_weighted


def compute_l_dot_n(points_hit, foreground_mask, background_mask, surface_normals_hit_unit, point_lights_position):
    point_light_position = point_lights_position[0]
    # number of intersections
    n_points_hit = points_hit.shape[0]
    # make a tensor that contains one copy of the point light position per intersection
    point_light_position_per_point_hit = point_light_position.unsqueeze(0).repeat(n_points_hit, 1)
    assert is_float_tensor_on_device(point_light_position_per_point_hit)
    assert point_light_position_per_point_hit.shape == (n_points_hit, 3)
    # compute the direction vectors from intersections to point light position
    point_light_rays = point_light_position_per_point_hit - points_hit
    point_light_rays[background_mask] = zero_dot_zero
    point_light_rays_unit = tf.normalize(point_light_rays)
    point_light_rays_unit[background_mask] = zero_dot_zero
    # sanity check
    point_light_rays_unit_norm = torch.norm(point_light_rays_unit, p=2, dim=1, keepdim=True)
    assert is_float_tensor_on_device(point_light_rays_unit_norm)
    assert torch.allclose(point_light_rays_unit_norm[foreground_mask], one_dot_zero)
    assert torch.allclose(point_light_rays_unit_norm[background_mask], zero_vector_float)
    # compute dot products
    l_dot_n = torch.sum(torch.mul(surface_normals_hit_unit, point_light_rays_unit), dim=1)
    assert is_float_tensor_on_device(l_dot_n)
    assert l_dot_n.shape == (n_points_hit,)
    return l_dot_n, point_light_rays, point_light_rays_unit
