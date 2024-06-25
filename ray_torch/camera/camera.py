import torch
import torch.nn.functional as tf
import torch.linalg as la

from ray_torch.constant.constant import minus_one_dot_zero
from ray_torch.constant.constant import one_dot_zero
from ray_torch.constant.constant import two_dot_zero
from ray_torch.utility.utility import create_tensor_on_device
from ray_torch.utility.utility import device
from ray_torch.utility.utility import is_float_tensor_on_device
from ray_torch.utility.utility import log_level_debug
from ray_torch.utility.utility import normalize_vector
from ray_torch.utility.utility import normalize_vector_custom
from ray_torch.utility.utility import see

near = create_tensor_on_device(0.1)
far = create_tensor_on_device(100.0)

resx_float_py = 1920.0
resy_float_py = 1080.0
resx_int_py = int(resx_float_py)
resy_int_py = int(resy_float_py)
resx = create_tensor_on_device(resx_float_py)
resy = create_tensor_on_device(resy_float_py)

middle_pixel_index = int((resy_int_py / 2) * resx_int_py + (resy_int_py / 2))

n_pixels = int((resx * resy).item())
see("n_pixels", n_pixels)
assert isinstance(n_pixels, int)

resx_half = resx / two_dot_zero
resy_half = resy / two_dot_zero

see("far", far)
see("near", near)
see("resx", resx)
see("resy", resy)
see("resx_half", resx_half)
see("resy_half", resy_half)

# position of the eye point
# eye = create_tensor_on_device([0.22, 0.0, -0.44])
eye = create_tensor_on_device([0.0, 0.0, 0.0])
see("eye", eye)
see("eye.type", eye.type())
see("eye.device", eye.device)
assert is_float_tensor_on_device(eye)

# upright direction of the camera orientation
up = create_tensor_on_device([0.0, 1.0, 0.0])
up = tf.normalize(up, dim=0)
assert torch.allclose(up, normalize_vector_custom(up))
assert torch.allclose(up, normalize_vector(up))
see("up", up)
assert is_float_tensor_on_device(up)

# look is center of image plane
# look = create_tensor_on_device([1.0, 0.0, 2.0])
look = create_tensor_on_device([0.0, 0.0, 1.5])
see("look", look)
assert is_float_tensor_on_device(look)

gaze = look - eye
assert is_float_tensor_on_device(gaze)

# distance from the eye to center of image plane
distance_intrinsic = torch.norm(gaze)
see("distance_intrinsic", distance_intrinsic)
assert is_float_tensor_on_device(distance_intrinsic)

# direction from eye towards center of image plane
gaze_unit = tf.normalize(gaze, dim=0)
see("gaze", gaze)
see("gaze_unit", gaze_unit)
assert torch.isclose(torch.norm(gaze_unit), torch.tensor(1., dtype=torch.float, requires_grad=False))
assert is_float_tensor_on_device(gaze_unit)

scrnx_unit = la.cross(up, gaze_unit, dim=0)
scrnx_unit = tf.normalize(scrnx_unit, dim=0)
assert torch.isclose(torch.norm(scrnx_unit), torch.tensor(1., dtype=torch.float, requires_grad=False))
# !? hack to get numerically perfect scrnx in special case
scrnx_unit_perfect = create_tensor_on_device([1.,0.,0.])
if torch.allclose(scrnx_unit, scrnx_unit_perfect):
    print("Using perfect scrnx_unit.")
    scrnx_unit = scrnx_unit_perfect
see("scrnx_unit", scrnx_unit)
assert is_float_tensor_on_device(scrnx_unit)

scrny_unit = la.cross(gaze_unit, scrnx_unit, dim=0)
scrny_unit = tf.normalize(scrny_unit, dim=0)
assert torch.isclose(torch.norm(scrny_unit), torch.tensor(1., dtype=torch.float, requires_grad=False))
# !? hack to get numerically perfect scrny in special case
scrny_unit_perfect = create_tensor_on_device([0.,1.,0.])
if torch.allclose(scrny_unit, scrny_unit_perfect):
    print("Using perfect scrny_unit.")
    scrny_unit = scrny_unit_perfect
see("scrny_unit", scrny_unit)
assert is_float_tensor_on_device(scrny_unit)

# !? note that we compute scrnz so it points towards the eye
# i.e. it is the most reasonable normal of the image plane
scrnz_unit = la.cross(scrnx_unit, scrny_unit, dim=0)
scrnz_unit = tf.normalize(scrnz_unit, dim=0)
# !? hack to get numerically perfect scrnz in special case
scrnz_unit_perfect = create_tensor_on_device([0.,0.,1.])
if torch.allclose(scrnz_unit, scrnz_unit_perfect):
    print("Using perfect scrnz_unit.")
    scrnz_unit = scrnz_unit_perfect
see("scrnz_unit", scrnz_unit)
assert is_float_tensor_on_device(scrnz_unit)

# note that fovx is actually representing half of the horizontal field of view
fovx_degrees = create_tensor_on_device(50.0)
see("fovx_degrees", fovx_degrees)
assert is_float_tensor_on_device(fovx_degrees)

fovx_radians = torch.deg2rad(fovx_degrees)
see("fovx_radians", fovx_radians)
assert is_float_tensor_on_device(fovx_radians)

# note that fovy is actually representing half of the vertical field of view
fovy_degrees = fovx_degrees / (resx / resy)
# resx and resy are already tensors thus no neeed to wrap resx / resy
# fovy_degrees = fovx_degrees / torch.tensor(resx / resy, dtype=torch.float)
see("fovy_degrees", fovy_degrees)
assert is_float_tensor_on_device(fovy_degrees)

# override fovy to optimize magnitude of a pixel in y direction
fovy_degrees = create_tensor_on_device(33.83)
see("fovy_degrees", fovy_degrees)
assert is_float_tensor_on_device(fovy_degrees)

fovy_radians = torch.deg2rad(fovy_degrees)
see("fovy_radians", fovy_radians)
assert is_float_tensor_on_device(fovy_radians)

# compute size of a pixel and scale scrnx and scrny accordingly


def compute_magnitude_of_a_pixel(distance, fov_radians, res, two_dot_zero):
    mag = torch.abs(two_dot_zero * distance * (torch.tan(fov_radians) / res))
    assert is_float_tensor_on_device(mag)
    return mag


# magx is the length aka. magnitude of a pixel in direction scrnx
magx = compute_magnitude_of_a_pixel(distance_intrinsic, fovx_radians, resx, two_dot_zero)
scrnx_scaled = scrnx_unit * magx
# scrnx is now 1 pixel long in horizontal direction of the image plane
see("scrnx_scaled", scrnx_scaled)
assert torch.isclose(torch.norm(scrnx_scaled), magx)
assert is_float_tensor_on_device(magx)
assert is_float_tensor_on_device(scrnx_scaled)

# magy is the length aka. magnitude of a pixel in direction scrny
magy = compute_magnitude_of_a_pixel(distance_intrinsic, fovy_radians, resy, two_dot_zero)
scrny_scaled = scrny_unit * magy
# scrny is now 1 pixel long in vertical direction of the image plane
see("scrny_scaled", scrny_scaled)
assert torch.isclose(torch.norm(scrny_scaled), magy)
assert is_float_tensor_on_device(magy)
assert is_float_tensor_on_device(scrny_scaled)

# compute position of the pixel in the upper left corner of the image plane relative to the eye


def scale_scrn_vectors(scrnx, resx, resx_off, scrny, resy, resy_off, two_dot_zero):
    ox = scrnx * ((resx / two_dot_zero) - resx_off)
    oy = scrny * ((resy / two_dot_zero) - resy_off)
    return ox, oy


def compute_relative_position_of_pixel_upper_left(gaze, scrnx, resx, resx_off, scrny, resy, resy_off, two_dot_zero):
    # we assume that gaze is pointing exactly to the middle of the image plane
    # i.e. gaze is pointing at the center of the point between the four mid-most pixels
    ox, oy = scale_scrn_vectors(scrnx, resx, resx_off, scrny, resy, resy_off, two_dot_zero)
    px_ul_pos = gaze - ox + oy
    assert is_float_tensor_on_device(px_ul_pos)
    return px_ul_pos

# we also compute these relative positions of some other pixels just for sanity checks


def compute_relative_position_of_pixel_lower_left(gaze, scrnx, resx, resx_off, scrny, resy, resy_off, two_dot_zero):
    # we assume that gaze is pointing exactly to the middle of the image plane
    # i.e. gaze is pointing at the center of the point between the four mid-most pixels
    ox, oy = scale_scrn_vectors(scrnx, resx, resx_off, scrny, resy, resy_off, two_dot_zero)
    px_ll_pos = gaze - ox - oy
    assert is_float_tensor_on_device(px_ll_pos)
    return px_ll_pos


def compute_relative_position_of_pixel_upper_right(gaze, scrnx, resx, resx_off, scrny, resy, resy_off, two_dot_zero):
    # we assume that gaze is pointing exactly to the middle of the image plane
    # i.e. gaze is pointing at the center of the point between the four mid-most pixels
    ox, oy = scale_scrn_vectors(scrnx, resx, resx_off, scrny, resy, resy_off, two_dot_zero)
    px_ur_pos = gaze + ox + oy
    assert is_float_tensor_on_device(px_ur_pos)
    return px_ur_pos


def compute_relative_position_of_pixel_lower_right(gaze, scrnx, resx, resx_off, scrny, resy, resy_off, two_dot_zero):
    # we assume that gaze is pointing exactly to the middle of the image plane
    # i.e. gaze is pointing at the center of the point between the four mid-most pixels
    ox, oy = scale_scrn_vectors(scrnx, resx, resx_off, scrny, resy, resy_off, two_dot_zero)
    px_lr_pos = gaze + ox - oy
    assert is_float_tensor_on_device(px_lr_pos)
    return px_lr_pos


# factors to ensure computed positions are the middle of pixels
resx_off = create_tensor_on_device(0.5)
resy_off = create_tensor_on_device(0.5)
assert is_float_tensor_on_device(resx_off)
assert is_float_tensor_on_device(resy_off)

px_ul = compute_relative_position_of_pixel_upper_left(gaze, scrnx_scaled, resx, resx_off, scrny_scaled, resy, resy_off, two_dot_zero)
see("px_ul", px_ul)

px_ll = compute_relative_position_of_pixel_lower_left(gaze, scrnx_scaled, resx, resx_off, scrny_scaled, resy, resy_off, two_dot_zero)
see("px_ll", px_ll)

px_ur = compute_relative_position_of_pixel_upper_right(gaze, scrnx_scaled, resx, resx_off, scrny_scaled, resy, resy_off, two_dot_zero)
see("px_ur", px_ur)

px_lr = compute_relative_position_of_pixel_lower_right(gaze, scrnx_scaled, resx, resx_off, scrny_scaled, resy, resy_off, two_dot_zero)
see("px_lr", px_lr)

distance_px_ul_px_ur = torch.norm(px_ul - px_ur)
see("distance_px_ul_px_ur", distance_px_ul_px_ur)

# note that a pixel position vector points to the middle of a pixel
# therefore when calculating a width of the image plane
# we count two times half a pixel less
# which is represented here by subtracting resx_off twice before scaling
distance_px_ul_px_ur_anticipated = (resx - resx_off - resx_off) * magx
see("distance_px_ul_px_ur_anticipated", distance_px_ul_px_ur_anticipated, False)

assert torch.isclose(distance_px_ul_px_ur, distance_px_ul_px_ur_anticipated)

distance_px_ul_px_ll = torch.norm(px_ul - px_ll)
see("distance_px_ul_px_ll", distance_px_ul_px_ll, False)

# note that a pixel position vector points to the middle of a pixel
# therefore when calculating a height of the image plane
# we count two times half a pixel less
# which is represented here by subtracting resy_off twice before scaling
distance_px_ul_px_ll_anticipated = (resy - resy_off - resy_off) * magy
see("distance_px_ul_px_ll_anticipated", distance_px_ul_px_ll_anticipated, False)

assert torch.isclose(distance_px_ul_px_ll, distance_px_ul_px_ll_anticipated)

distance_aspect_ratio = distance_px_ul_px_ur / distance_px_ul_px_ll
see("distance_aspect_ratio", distance_aspect_ratio, False)

distance_aspect_ratio_anticipated = distance_px_ul_px_ur / distance_px_ul_px_ll
see("distance_aspect_ratio_anticipated", distance_aspect_ratio_anticipated, False)

if fovy_degrees.item() == 33.83:
    distance_aspect_ratio_anticipated_also = resx / resy
    see("distance_aspect_ratio_anticipated_also", distance_aspect_ratio_anticipated_also, False)

    assert torch.isclose(distance_aspect_ratio, distance_aspect_ratio_anticipated_also, rtol=1e-03)

    distance_aspect_ratio_anticipated_moreover = create_tensor_on_device(16.0 / 9.0)
    see("distance_aspect_ratio_anticipated_moreover", distance_aspect_ratio_anticipated_moreover, False)

    assert torch.isclose(distance_aspect_ratio, distance_aspect_ratio_anticipated_moreover, rtol=1e-03)

# compute the grid of vectors that form the image plane

# each pixel will be represented by a vector pointing to its middle

# create a vector per pixel on the image plane
# with each pixel 1.0 wide and 1.0 high
# and with the center of the image plane is the origin

img_grid_original_firstx = minus_one_dot_zero * ((resx / two_dot_zero) - resx_off)
img_grid_original_lastx = ((resx / two_dot_zero) - resx_off + one_dot_zero)
img_grid_original_firsty =((resy / two_dot_zero) - resy_off)
img_grid_original_lasty = minus_one_dot_zero * ((resy / two_dot_zero) - resy_off + one_dot_zero)

img_grid_original = torch.cartesian_prod(torch.arange(start=img_grid_original_firstx, end=img_grid_original_lastx, step=1.0, dtype=torch.float, requires_grad=False),
                                         torch.arange(start=img_grid_original_firsty, end=img_grid_original_lasty, step=-1.0, dtype=torch.float, requires_grad=False),
                                         torch.tensor([0.0], dtype=torch.float, requires_grad=False))
see("img_grid_original", img_grid_original)
img_grid_original = img_grid_original.to(device)
assert is_float_tensor_on_device(img_grid_original)

n_pixels_anticipated = img_grid_original.shape[0]
assert isinstance(n_pixels_anticipated, int)
assert n_pixels == n_pixels_anticipated

# scale the image plane using the width of a pixel and the height of a pixel

identity_matrix_3_by_3 = torch.eye(3, dtype=torch.float, requires_grad=False).to(device)
img_grid_scaling_matrix = identity_matrix_3_by_3 * create_tensor_on_device([magx, magy, 0.0])
see("img_grid_scaling_matrix", img_grid_scaling_matrix, False)
assert is_float_tensor_on_device(img_grid_scaling_matrix)

img_grid_scaled = torch.matmul(img_grid_original, img_grid_scaling_matrix)
see("img_grid_scaled", img_grid_scaled, False)
assert is_float_tensor_on_device(img_grid_scaled)

distance_px_ul_px_lr = torch.norm(px_ul - px_lr)
see("distance_px_ul_px_lr", distance_px_ul_px_lr, False)

distance_px_ul_px_lr_after_scaling = torch.norm(img_grid_scaled[n_pixels - 1] - img_grid_scaled[0])
see("distance_px_ul_px_lr_after_scaling", distance_px_ul_px_lr_after_scaling, False)

assert is_float_tensor_on_device(distance_px_ul_px_lr_after_scaling)
assert torch.isclose(distance_px_ul_px_lr, distance_px_ul_px_lr_after_scaling)

# rotate the image plane around the origin

scrnz_unit_neg = normalize_vector(minus_one_dot_zero * scrnz_unit)
assert is_float_tensor_on_device(scrnz_unit_neg)

img_grid_rotation_matrix = torch.stack([scrnx_unit, scrny_unit, scrnz_unit_neg], dim=1)
#img_grid_rotation_matrix = torch.stack([scrnx_unit, scrny_unit, scrnz_unit], dim=1)
assert is_float_tensor_on_device(img_grid_rotation_matrix)
see("img_grid_rotation_matrix", img_grid_rotation_matrix, False)

img_grid_rotated_px_ul = img_grid_rotation_matrix @ img_grid_scaled[0]
see("img_grid_rotated_px_ul", img_grid_rotated_px_ul, False)

img_grid_rotated_px_lr = img_grid_rotation_matrix @ img_grid_scaled[n_pixels - 1]
see("img_grid_rotated_px_lr", img_grid_rotated_px_lr, False)

img_grid_rotated = torch.matmul(img_grid_scaled, img_grid_rotation_matrix)
see("img_grid_rotated", img_grid_rotated, False)

distance_px_ul_px_lr_after_rotation = torch.norm(img_grid_rotated[n_pixels - 1] - img_grid_rotated[0])
see("distance_px_ul_px_lr_after_rotation", distance_px_ul_px_lr_after_rotation, False)

assert torch.isclose(distance_px_ul_px_lr_after_scaling, distance_px_ul_px_lr_after_rotation)
assert torch.isclose(distance_px_ul_px_lr, distance_px_ul_px_lr_after_rotation)

# then we translate the image plane into position

# translage by look

img_grid_translated_by_look = img_grid_rotated + look
see("img_grid_translated_by_look", img_grid_translated_by_look, False)

distance_px_ul_px_lr_after_translation_by_look = torch.norm(img_grid_translated_by_look[n_pixels - 1] - img_grid_translated_by_look[0])
see("distance_px_ul_px_lr_after_translation_by_look", distance_px_ul_px_lr_after_translation_by_look, False)

assert torch.isclose(distance_px_ul_px_lr_after_scaling, distance_px_ul_px_lr_after_translation_by_look)
assert torch.isclose(distance_px_ul_px_lr_after_scaling, distance_px_ul_px_lr_after_translation_by_look)
assert torch.isclose(distance_px_ul_px_lr, distance_px_ul_px_lr_after_translation_by_look)

# translate by eye and gaze

img_grid_translated_by_eye_and_gaze = img_grid_rotated + eye + gaze
see("img_grid_translated_by_eye_and_gaze", img_grid_translated_by_eye_and_gaze, False)

distance_px_ul_px_lr_after_translation_by_eye_and_gaze = torch.norm(img_grid_translated_by_eye_and_gaze[n_pixels - 1] - img_grid_translated_by_eye_and_gaze[0])
see("distance_px_ul_px_lr_after_translation_by_eye_and_gaze", distance_px_ul_px_lr_after_translation_by_eye_and_gaze, False)

assert torch.isclose(distance_px_ul_px_lr_after_scaling, distance_px_ul_px_lr_after_translation_by_eye_and_gaze)
assert torch.isclose(distance_px_ul_px_lr_after_scaling, distance_px_ul_px_lr_after_translation_by_eye_and_gaze)
assert torch.isclose(distance_px_ul_px_lr, distance_px_ul_px_lr_after_translation_by_eye_and_gaze)

# as expected these translations result in the same grid

assert torch.allclose(img_grid_translated_by_look, img_grid_translated_by_eye_and_gaze, rtol=0.1)

img_grid = img_grid_translated_by_look
assert is_float_tensor_on_device(img_grid)
see("img_grid", img_grid)

# compute a direction vector per primary ray

# since we have the image grid
# this is now just one parallelized operation


if log_level_debug:
    print(f"eye={eye}")
    print(f"look={look}")
    print(f"gaze={gaze}")
    print(f"up={up}")

    print(f"scrnx_unit={scrnx_unit}")
    print(f"scrny_unit={scrny_unit}")
    print(f"scrnz_unit={scrnz_unit}")

    print(f"scrnx_unit dot scrny_unit={torch.dot(scrnx_unit, scrny_unit)}")
    print(f"scrnx_unit dot scrnz_unit={torch.dot(scrnx_unit, scrnz_unit)}")
    print(f"scrny_unit dot scrnz_unit={torch.dot(scrny_unit, scrnz_unit)}")

    print(f"scrnx_scaled dot scrny_scaled={torch.dot(scrnx_scaled, scrny_scaled)}")
    print(f"scrnx_scaled dot scrnz_unit={torch.dot(scrnx_scaled, scrnz_unit)}")
    print(f"scrny_scaled dot scrnz_unit={torch.dot(scrny_scaled, scrnz_unit)}")

    print(f"scrnx_unit dot gaze={torch.dot(scrnx_unit, gaze)}")
    print(f"scrny_unit dot gaze={torch.dot(scrny_unit, gaze)}")

    print(f"gaze_unit dot scrnz_unit={torch.dot(gaze_unit, scrnz_unit)}")
    print(f"gaze_unit dot up={torch.dot(gaze_unit, up)}")

    print(f"gaze dot up={torch.dot(gaze, up)}")

    # print(f" dot ={torch.dot(torch.tensor([1.0, -1.0, -1.0], dtype=float, requires_grad=False), torch.tensor([1.0, 1.0, -1.0], dtype=float, requires_grad=False))}")
    # print(f" dot ={torch.dot(torch.tensor([1.0, -1.0, 1.0], dtype=float, requires_grad=False), torch.tensor([1.0, 1.0, 1.0], dtype=float, requires_grad=False))}")
    # print(f" dot ={torch.dot(torch.tensor([1.0, -1.0, 0.0], dtype=float, requires_grad=False), torch.tensor([1.0, 1.0, 0.0], dtype=float, requires_grad=False))}")
    # print(f" dot ={torch.dot(torch.tensor([1.0, -1.0, -1.0], dtype=float, requires_grad=False), torch.tensor([1.0, 1.0, 0.0], dtype=float, requires_grad=False))}")

    print(f"img_grid=\n{img_grid}")
    print(f"eye={eye}")
