import torch
import math
# from diff_gaussian_rasterization import (GaussianRasterizationSettings, GaussianRasterizer)
from diff_gaussian_rasterization_polymask import (GaussianRasterizationSettings, GaussianRasterizer)

from .sh_utils import eval_sh


from gaussian_render.gsparams import GaussianModel
from gaussian_render.cameras import Camera

import pdb 

def render(viewpoint_camera: Camera, pc : GaussianModel, scaling_modifier = 1.0, ray_dists = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0

    # here we dont want to retain the gradients
    # try:
    #     screenspace_points.retain_grad()
    # except:
    #     pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=pc.get_bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    scales = pc.get_scaling
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # colors_precomp = None
    # shs = pc.get_features

    colors_precomp = pc.get_features
    shs = None
    # # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
    #     # gt_mask = gt_mask,
    #     means3D = means3D,
    #     means2D = means2D,
    #     shs = shs,
    #     colors_precomp = colors_precomp,
    #     opacities = opacity,
    #     scales = scales,
    #     rotations = rotations,
    #     cov3D_precomp = cov3D_precomp)

    # # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # # They will be excluded from value updates used in the splitting criteria.
    # return {"render": rendered_image,
    #         "viewspace_points": screenspace_points,
    #         "visibility_filter" : radii > 0,
    #         "radii": radii,
    #         "alpha": rendered_alpha, 
    #         "depth": rendered_depth}

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, pred_depth, pred_alpha, pred_xys, gt_dists = rasterizer(
        # gt_mask = gt_mask,
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        ray_dists=ray_dists)

    image_center = torch.tensor([(viewpoint_camera.image_width - 1) / 2, (viewpoint_camera.image_height - 1) / 2], device=pred_xys.device)[None, :]
    pred_dists = ((pred_xys - image_center) ** 2).sum(dim=-1) ** 0.5 # L2 distance,
    max_dist = (image_center ** 2).sum(dim=-1) ** 0.5

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "pred_dists": pred_dists / max_dist,
            "pred_alpha": pred_alpha,
            "gt_dists": gt_dists / max_dist}
