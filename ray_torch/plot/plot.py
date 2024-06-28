import matplotlib as mpl
import matplotlib.pyplot as plt
import pyvista as pv
import torch

from ray_torch.constant.constant import zero_vector_int


def plot_rgb_image(img_rgb, background_mask, resx_int_py, resy_int_py):
    img_rgb = img_rgb.clone()
    img_rgb[background_mask] = zero_vector_int
    img_rgb_view = img_rgb.view(resx_int_py, resy_int_py, 3)
    img_rgb_view_permuted = img_rgb_view.permute(1, 0, 2)
    assert img_rgb_view_permuted.shape == (resy_int_py, resx_int_py, 3)
    plt.imshow(img_rgb_view_permuted, resample=False)
    plt.show()


def plot_rgb_image_with_actual_size(img_rgb, background_mask, resx_int_py, resy_int_py):
    img_rgb = img_rgb.clone()
    img_rgb[background_mask] = zero_vector_int
    img_rgb_view = img_rgb.view(resx_int_py, resy_int_py, 3)
    img_rgb_view_permuted = img_rgb_view.permute(1, 0, 2)
    assert img_rgb_view_permuted.shape == (resy_int_py, resx_int_py, 3)
    # get DPI
    dpi = mpl.rcParams["figure.dpi"]
    # what size does the figure need to be in inches to fit the image
    figsize = resx_int_py / float(dpi), resy_int_py / float(dpi)
    # create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    # hide spines, ticks, etc.
    ax.axis("off")
    ax.imshow(img_rgb_view_permuted)
    plt.show()


def plot_vectors_with_color_by_z_value(vectors, foreground_mask, clim, cmap, show_axes=True, show_grid=True):
    pcd_np = vectors[foreground_mask].numpy()
    pcd_pv = pv.PolyData(pcd_np)
    pcd_pv["point_color"] = pcd_pv.points[:, 2]  # use z-coordinate as color
    pv.plot(
        pcd_pv,
        scalars="point_color",  # use z-coordinate as color
        clim=clim,
        cmap=cmap,
        show_axes=show_axes,
        show_grid=show_grid,
        parallel_projection=True,
        render_points_as_spheres=False,
    )


def plot_vectors_with_color_by_norm(vectors, foreground_mask, clim, cmap, show_axes=True, show_grid=True):
    pcd_pv = pv.PolyData(vectors[foreground_mask].numpy())
    pcd_pv["point_color"] = torch.norm(vectors[foreground_mask], p=2, dim=1, keepdim=True).numpy()
    pv.plot(
        pcd_pv,
        scalars="point_color",  # use L2 norm as color
        clim=clim,
        cmap=cmap,
        show_axes=show_axes,
        show_grid=show_grid,
        parallel_projection=True,
        render_points_as_spheres=False,
    )
