import pyvista as pv
import torch


def plot_vectors_with_color_by_z_value(vectors, foreground_mask, clim, cmap, show_axes=True, show_grid=True):
    pcd_np = vectors[foreground_mask].numpy()
    pcd_pv = pv.PolyData(pcd_np)
    pcd_pv['point_color'] = pcd_pv.points[:, 2]  # use z-coordinate as color
    pv.plot(pcd_pv,
            scalars='point_color',  # use z-coordinate as color
            clim=clim,
            cmap=cmap,
            show_axes=show_axes,
            show_grid=show_grid,
            parallel_projection=True,
            render_points_as_spheres=False)


def plot_vectors_with_color_by_norm(vectors, foreground_mask, clim, cmap, show_axes=True, show_grid=True):
    pcd_pv = pv.PolyData(vectors[foreground_mask].numpy())
    pcd_pv['point_color'] = torch.norm(vectors[foreground_mask], p=2, dim=1, keepdim=True).numpy()
    pv.plot(pcd_pv,
            scalars='point_color',  # use L2 norm as color
            clim=clim,
            cmap=cmap,
            show_axes=show_axes,
            show_grid=show_grid,
            parallel_projection=True,
            render_points_as_spheres=False)
