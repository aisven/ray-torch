import torch
import torch.nn.functional as tf

from ray_torch.camera.camera import far
from ray_torch.constant.constant import four_dot_zero
from ray_torch.constant.constant import minus_zero_dot_five
from ray_torch.constant.constant import one_dot_zero
from ray_torch.constant.constant import two_dot_zero
from ray_torch.constant.constant import zero_vector_float
from ray_torch.utility.utility import is_float_tensor_on_device
from ray_torch.utility.utility import is_long_tensor
from ray_torch.utility.utility import see
from ray_torch.utility.utility import see_more


def intersect_rays_with_spheres(n_rays, n_spheres, ray_origin_per_sphere, spheres_center, spheres_radius, primary_ray_vectors_unit):
    # n_rays is the number of rays

    # n_spheres is the number of spheres

    # ray_origin_per_sphere contains one row per sphere where each row is the 3D vector of the origin of the ray
    # when shooting rays from the camera this is an n by 3 tensor where each row is the eye vector
    assert ray_origin_per_sphere.shape == (n_spheres, 3)

    # spheres_center contains one row per sphere where each row is the 3D vector if the center of the sphere
    assert spheres_center.shape == (n_spheres, 3)

    # spheres_radius contains one scalar per sphere where each scalar is the radius of the sphere in radians
    assert spheres_radius.shape == (n_spheres,)

    # notation somewhat aligned with the lecture script
    # E = P = eye = where all primary rays start
    # D = primary ray = direction from eye to pixel represented as unit vector
    # S = center of a sphere
    # r = radius of a sphere
    # t = distance from eye to intersection point
    # R = E + t * D = vector of intersection point

    # compute the coefficient A of the quadratic equation
    # A = x_D^2 + y_D^2 + z_D^2 = dot(D, D)
    # which is just the dot product of D with itself
    # which is 1.0 since the ray direction vectors are unit vectors
    a = one_dot_zero

    # compute the coefficient b of the quadratic equation
    # B = 2 * (x_D(x_E - x_S) + y_D(y_E - y_S) + z_D(z_E - z_S)) = 2 * dot(D, E - S)
    # compute E - C for all rays with one element-wise matrix subtraction
    ray_origin_minus_spheres_center = ray_origin_per_sphere - spheres_center
    see("ray_origin_minus_spheres_center", ray_origin_minus_spheres_center, False)
    assert is_float_tensor_on_device(ray_origin_minus_spheres_center)
    assert ray_origin_minus_spheres_center.shape == (n_spheres, 3)
    # we actually compute all dot products with just one matrix multiplication
    b = two_dot_zero * torch.matmul(primary_ray_vectors_unit, ray_origin_minus_spheres_center.T)
    see("b", b)
    assert is_float_tensor_on_device(b)
    assert b.shape == (n_rays, n_spheres)

    # compute the coefficient c of the quadratic equation
    # C = (x_E - x_S)^2 + (y_E - y_S)^2 + (z_E - z_S)^2 = dot(E - S, E - S)

    # compute the square of each radius
    spheres_radius_sqaured = torch.square(spheres_radius)
    see("spheres_radius_sqaured", spheres_radius_sqaured, False)
    assert is_float_tensor_on_device(spheres_radius_sqaured)
    assert spheres_radius_sqaured.shape == (n_spheres,)
    # we compute all the dot products and all the scalar subtractions in one go
    c = torch.sum(torch.mul(ray_origin_minus_spheres_center, ray_origin_minus_spheres_center),
                  dim=1) - spheres_radius_sqaured
    see("c", c)
    assert is_float_tensor_on_device(c)
    assert c.shape == (n_spheres,)

    # compute the discriminant of the quadratic equation
    # discriminant = B^2 - 4 * C
    four_dot_zero_times_c_stacked = (four_dot_zero * c).unsqueeze(0).repeat(n_rays, 1)
    see("four_dot_zero_times_c_stacked", four_dot_zero_times_c_stacked, False)
    assert is_float_tensor_on_device(four_dot_zero_times_c_stacked)
    assert four_dot_zero_times_c_stacked.shape == (n_rays, n_spheres)
    discriminants = torch.square(b) - four_dot_zero_times_c_stacked
    see("discriminants", discriminants, False)
    assert is_float_tensor_on_device(discriminants)
    assert discriminants.shape == (n_rays, n_spheres)

    # mask discriminants regarding number of intersections between any ray and sphere
    spheres_2_solution_indices = discriminants > 1e-8
    see("spheres_2_solution_indices", spheres_2_solution_indices, False)
    assert spheres_2_solution_indices.shape == (n_rays, n_spheres)
    spheres_1_solution_indices = (discriminants > -1e-8) & (discriminants < 1e-8)
    see("spheres_1_solution_indices", spheres_1_solution_indices, False)
    assert spheres_1_solution_indices.shape == (n_rays, n_spheres)
    spheres_0_solution_indices = (discriminants < 1e-8)
    see("spheres_0_solution_indices", spheres_0_solution_indices, False)
    assert spheres_0_solution_indices.shape == (n_rays, n_spheres)

    # count for how many pairs of rays spheres there are 2, 1, 0 solutions
    n_2_solutions = torch.count_nonzero(spheres_2_solution_indices)
    see("n_2_solutions", n_2_solutions)
    n_1_solutions = torch.count_nonzero(spheres_1_solution_indices)
    see("n_1_solutions", n_1_solutions)
    n_0_solutions = torch.count_nonzero(spheres_0_solution_indices)
    see("n_0_solutions", n_0_solutions)

    # compute the square root of the discriminant of the quadratic equation
    # sqrt(discriminant) = sqrt(B^2 - 4 * C)
    # note that there are two solutions to the square root if discriminant is > 0
    # we accomodate for that by using + and - sign in subsequent formula
    discriminants_sqrt = discriminants.clone()
    discriminants_sqrt[spheres_2_solution_indices] = torch.sqrt(discriminants[spheres_2_solution_indices])
    discriminants_sqrt[spheres_1_solution_indices] = 0.0
    discriminants_sqrt[spheres_0_solution_indices] = -1.0
    see("discriminants_sqrt", discriminants_sqrt, False)
    assert is_float_tensor_on_device(discriminants_sqrt)
    assert discriminants_sqrt.shape == (n_rays, n_spheres)

    # compute the distances from the eye to the intersections
    # t_0 = (- B - sqrt(B^2 - 4 * C)) / 2 = -0.5 * (B + sqrt(B^2 - 4 * C))
    # t_1 = (- B + sqrt(B^2 - 4 * C)) / 2 = -0.5 * (B - sqrt(B^2 - 4 * C))
    t_0s = minus_zero_dot_five * (b + discriminants_sqrt)
    t_0s[spheres_0_solution_indices] = 0.0
    see_more("t_0s", t_0s, False)
    assert is_float_tensor_on_device(t_0s)
    assert t_0s.shape == (n_rays, n_spheres)
    t_1s = minus_zero_dot_five * (b - discriminants_sqrt)
    t_1s[spheres_0_solution_indices] = 0.0
    see_more("t_0s", t_0s, False)
    see_more("t_1s", t_1s, False)
    assert is_float_tensor_on_device(t_1s)
    assert t_1s.shape == (n_rays, n_spheres)

    # note that in case a sphere would be completely or partially behind the camera
    # we would need to cull away all intersections behind the camera
    # by setting them to far or infinity before taking the minimum

    # note that the 1 solution case is rare
    # and corresponding values in t_0s and t_1s equal
    # in that case the minimum function will simply use that value

    ts = torch.minimum(t_0s, t_1s)
    ts[spheres_0_solution_indices] = 0.0
    see_more("ts", ts, False)
    assert is_float_tensor_on_device(ts)
    assert ts.shape == (n_rays, n_spheres)

    # determine the minimum t for each ray
    # ts[spheres_0_solution_indices] = far + 10.0
    ts[spheres_0_solution_indices] = far + 10.0
    ts_minimum = torch.min(ts, dim=1)
    ts_min = ts_minimum.values
    ts_background_mask = ts_min > far
    assert ts_background_mask.shape == (n_rays,)
    ts_foreground_mask = ts_min <= far
    assert ts_foreground_mask.shape == (n_rays,)
    ts_foreground_mask_with_0 = ts_foreground_mask.clone()
    ts_foreground_mask_with_0[0] = True
    ts_min[ts_background_mask] = 0.0
    see_more("ts_min", ts_min)
    assert is_float_tensor_on_device(ts_min)
    assert ts_min.shape == (n_rays,)

    # compute intersection points
    # by scaling each primary ray unit vector by the corresponding minimum t
    points_hit = torch.mul(primary_ray_vectors_unit, torch.unsqueeze(ts_min, dim=1))
    points_hit[ts_background_mask] = zero_vector_float
    see_more("points_hit", points_hit)
    assert points_hit.shape == (n_rays, 3)

    # for each ray note the index of the sphere that the ray hit
    # ts_min_spheres_index = torch.remainder(ts_minimum.indices, n_spheres)
    spheres_index_hit = torch.remainder(ts_minimum.indices, n_spheres)
    see_more("ts_min_spheres_index", spheres_index_hit)
    assert is_long_tensor(spheres_index_hit)
    assert ts_min.shape == (n_rays,)

    # for each ray get the center of the sphere that the ray hit
    spheres_center_hit = spheres_center[spheres_index_hit]
    see("spheres_center_hit", spheres_center_hit)
    assert is_float_tensor_on_device(spheres_center_hit)
    assert spheres_center_hit.shape == (n_rays, 3)

    # also compute surface normal at each intersection in terms of a unit vector
    surface_normals_hit = points_hit - spheres_center_hit
    surface_normals_hit[ts_background_mask] = zero_vector_float
    surface_normals_hit_unit = tf.normalize(surface_normals_hit)
    see_more("surface_normals_hit_unit", surface_normals_hit_unit)
    assert is_float_tensor_on_device(surface_normals_hit_unit)

    # sanity check
    surface_normals_hit_unit_norm = torch.norm(surface_normals_hit_unit, p=2, dim=1, keepdim=True)
    see_more("surface_normals_unit_norm", surface_normals_hit_unit_norm, False)
    assert is_float_tensor_on_device(surface_normals_hit_unit_norm)
    assert torch.allclose(surface_normals_hit_unit_norm[ts_foreground_mask], one_dot_zero)
    assert torch.allclose(surface_normals_hit_unit_norm[ts_background_mask], zero_vector_float)

    background_mask = ts_background_mask
    foreground_mask = ts_foreground_mask
    foreground_mask_with_0 = ts_foreground_mask_with_0

    return points_hit, surface_normals_hit, surface_normals_hit_unit, spheres_index_hit, spheres_center_hit, background_mask, foreground_mask, foreground_mask_with_0
