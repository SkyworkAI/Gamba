import math

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import pdb

from core.encoders.dinov2_wrapper import Dinov2Wrapper
from .gambaformer import GambaFormer, TriGambaFormer

from core.gsutils.typings import *
from core.options import Options
from core.gs import GaussianRenderer
from core.triplane_model import TriPlaneModel
from kiui.lpips import LPIPS



# from core.gsutils.ops import (get_activation, scale_tensor, trunc_exp)

inverse_sigmoid = lambda x: np.log(x / (1 - x))


class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_neurons: int,
        n_hidden_layers: int,
        activation: str = "relu",
        output_activation: Optional[str] = None,
        bias: bool = True,
    ):
        super().__init__()
        layers = [
            self.make_linear(
                dim_in, n_neurons, is_first=True, is_last=False, bias=bias
            ),
            self.make_activation(activation),
        ]
        for i in range(n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    n_neurons, n_neurons, is_first=False, is_last=False, bias=bias
                ),
                self.make_activation(activation),
            ]
        layers += [
            self.make_linear(
                n_neurons, dim_out, is_first=False, is_last=True, bias=bias
            )
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = self.make_activation(activation)

    def forward(self, x):
        x = self.layers(x)
        x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last, bias=True):
        layer = nn.Linear(dim_in, dim_out, bias=bias)
        return layer

    def make_activation(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        else:
            raise NotImplementedError


class CameraEmbedder(nn.Module):
    """
    Embed camera features to a high-dimensional vector.
    
    Reference:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L27
    """
    def __init__(self, raw_dim: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(raw_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class GSDecoder(nn.Module):
    def __init__(self, transformer_dim, 
                 SH_degree,
                 mlp_dim=128, 
                 init_density=0.1, 
                 clip_scaling=0.1,
                 enable_triplane=False,
                 tri_channels=1):
        super(GSDecoder, self).__init__()
        
        self.embed_dim = transformer_dim
        self.mlp_dim = mlp_dim
        self.clip_scaling = clip_scaling
        self.enable_triplane = enable_triplane

        self.mlp_net = MLP(transformer_dim, mlp_dim, n_neurons=128, n_hidden_layers=1, activation="silu")

        if self.enable_triplane:
            self.mlp_tri = MLP(tri_channels, mlp_dim, n_neurons=128, n_hidden_layers=5, activation="silu")

        pos_bound = 0.4
        coords = torch.linspace(-1 * pos_bound, pos_bound, 21)[None, :].repeat(3, 1)
        self.register_buffer("coords", coords)
        # xyz (3) + scale (3) + rot(4) + opacity(1) + SH_0 (3)
        # self.pred_keys = ["xyz", "scale", "rot", "opacity", "shs"]
        # align with lgm
        self.pred_keys = ["xyz", "opacity", "scale", "rot", "rgb"]

        # self.fix_keys = ["scale", "rot"]
        self.fix_keys = ["rot"]

        # self.tri_keys = ["rgb", "scale", "rot", "opacity"]
        self.tri_keys = ["rgb"]
        # self.fix_keys = []

        self.gs_layer = nn.ModuleDict()
        for key in self.pred_keys:
            if key in self.fix_keys:
                continue
            if key == "xyz":
                layer = nn.Linear(self.mlp_dim, 3 * self.coords.size(-1))
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.constant_(layer.bias,0)
            elif key == "scale":
                layer = nn.Linear(self.mlp_dim, 3)
                # nn.init.constant_(layer.bias, init_scaling)
                nn.init.constant_(layer.bias, -1.8)
                # nn.init.constant_(layer.bias, 0)
                # self.scale_min = 0.005
                # self.scale_max = 0.02
            elif key == "rot":
                layer = nn.Linear(self.mlp_dim, 4)
                nn.init.constant_(layer.bias, 0)
                # nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                layer = nn.Linear(self.mlp_dim, 1)
                nn.init.constant_(layer.bias, inverse_sigmoid(init_density))
            elif key == "shs":
                shs_dim = 3 * (SH_degree + 1) ** 2
                layer = nn.Linear(self.mlp_dim, int(shs_dim))
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, 0)
            elif key == "rgb":
                color_dim = 3
                layer = nn.Linear(self.mlp_dim, color_dim)
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, 0.0)
            else:
                raise NotImplementedError
            self.gs_layer[key] = layer

    def key_activation(self, v, key=''):
        if key == "xyz":
            # (bsz, num_pts, 3, prob_num)
            v = v.reshape(*v.shape[:2], 3, -1) 
            prob = F.normalize(v, dim=-1)
            # coords shape (1, 1, 3, prob_num)
            v = (prob * self.coords[None, None]).sum(dim=-1)
        elif key == "scale":
            # v = torch.sigmoid(v)
            # v = self.scale_min * v + (1 - v) * self.scale_max
            v = 0.1 * F.softplus(v)
        elif key == "rot":
            v = torch.nn.functional.normalize(v, dim=-1)
            # v = torch.nn.functional.normalize(v, dim=1)
        elif key == "opacity":
            v = torch.sigmoid(v)
        elif key == "shs":
            pass 
        elif key == "rgb":
            v = torch.sigmoid(v)
        else:
            raise NotImplementedError
        return v

    def forward_position(self, feats):
        assert self.enable_triplane
        feats = self.mlp_net(feats)
        output = {}
        for key in self.pred_keys:
            if key in self.tri_keys or key in self.fix_keys:
                continue
            v = self.gs_layer[key](feats)
            v = self.key_activation(v, key)
            output[key] = v
        return output

    def forward(self, feats, partial_output={}):
        gsparams = []
        if not self.enable_triplane:
            feats = self.mlp_net(feats)
        else:
            feats = self.mlp_tri(feats)
        
        for key in self.pred_keys:
            # skip fixed keys
            if key in self.fix_keys:
                if key == "scale":
                    fix_v = 0.03 * torch.ones(*feats.shape[:2], 3).to(feats.device)
                elif key == "rot":
                    fix_v = torch.zeros(*feats.shape[:2], 4).to(feats.device)
                    fix_v[:, :, 0] = 1.
                gsparams.append(fix_v)
                continue
            
            # skip keys has computed in forward_postion,
            if key in partial_output:
                gsparams.append(partial_output[key])
                continue
            
            v = self.gs_layer[key](feats)
            v = self.key_activation(v, key)
            gsparams.append(v)

        return torch.cat(gsparams, dim=-1)


class GSPredictor(nn.Module):
    def __init__(self, 
                 opt: Options,
                 SH_degree=0,
                 **kwargs):
        super().__init__()

        self.opt = opt
        self.enable_triplane = opt.use_triplane
        # self.map_camera = CameraEmbedder(
        #     raw_dim=12+4, embed_dim=opt.campose_dim,
        # )

        self.map_camera = nn.Identity()
                
        # default init
        self.initialize_weights()
        
        # consistent with train-sample.yaml in openlrm
        self.map_image = Dinov2Wrapper(model_name=opt.dino_name, freeze=True)

        self.stem_layer = nn.Conv2d(
            in_channels=9, out_channels=opt.gamba_dim, kernel_size=opt.patch_size, stride=opt.patch_size
        )

        self.transformer = GambaFormer(inner_dim=opt.gamba_dim, 
                                         image_feat_dim=opt.dino_dim, 
                                         mod_embed_dim=opt.campose_dim, 
                                         num_layers=opt.gamba_layers, 
                                         gs_num=opt.gs_num, 
                                         token_pnum=opt.token_pnum, drop_path_rate=0.1)
        if self.enable_triplane:
            self.triplane_processor = TriPlaneModel(opt)
            self.triplane_encoder = TriGambaFormer(inner_dim=opt.gamba_dim, 
                                         image_feat_dim=opt.dino_dim, 
                                         mod_embed_dim=opt.campose_dim, 
                                         num_layers=opt.gamba_layers, 
                                         drop_path_rate=0.1)
        tri_channels = 1 if not self.enable_triplane else self.triplane_processor.output_channels
        self.decoder = GSDecoder(opt.gamba_dim, SH_degree, enable_triplane=self.enable_triplane, tri_channels=tri_channels)
    
    def initialize_weights(self):
    # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize label embedding table:
        # nn.init.normal_(self.map_label.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        # nn.init.normal_(self.map_noise.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.map_noise.mlp[2].weight, std=0.02)
       
        # Initialize camera embedding MLP:
        # nn.init.normal_(self.map_camera.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.map_camera.mlp[2].weight, std=0.02)

    def forward(self, cond_views, cam_poses, plucker_img=None):
        """
        Input: noise views and class label with source camera pose 
        Output: clean gaussian splatting parameters 
        """
        
        # noise_views [N, C_img, H_img, W_img]
        # cam_pose: [N, D_cam_raw: 16]
        # class_lbl [N, ]
        # noise_lvl [N, ]
        
        # gs [N,gs_num, gs_dim:14]
        bsz, cond_num, c, h, w = cond_views.size()
        
        cond_views = cond_views.reshape(-1, c, h ,w).contiguous()
        img_cond = self.map_image(cond_views)

        plucker_cond = None
        if plucker_img is not None:
            bsz, cond_num, c, h, w = plucker_img.size()
            plucker_img = plucker_img.view(bsz*cond_num, c, h, w)
            plucker_cond = self.stem_layer(plucker_img) # [B*V, M, dim_e=768]
            p_cond_1 = plucker_cond.flatten(2)
            p_cond_2 = plucker_cond.permute(0, 1, 3, 2).flatten(2)
            p_cond_3 = plucker_cond.flip(dims=[3]).flatten(2)
            p_cond_4 = plucker_cond.permute(0, 1, 3, 2).flip(dims=[3]).flatten(2)
            # (bsz * view_num, C, token_num)
            plucker_cond = torch.cat([p_cond_1, p_cond_2, p_cond_3, p_cond_4], dim=-1)
            # (bsz, view_num * token_num, C)
            plucker_cond = plucker_cond.permute(0, 2, 1).reshape(bsz, -1, self.opt.gamba_dim)


        # c_cond = self.map_camera(cam_poses).view(bsz, -1)
        # mod = c_cond
        mod = None

        triplane_embedding = None if not self.enable_triplane else self.triplane_processor.get_embedding()

        net_out = self.transformer(img_cond, mod, plucker_cond=plucker_cond)

        if self.enable_triplane:
            tri_net_out = self.triplane_encoder(img_cond, mod, embedding=triplane_embedding)
        
        if not self.enable_triplane:
            gs = self.decoder(net_out["feats"])
            outputs = {"pred_gs": gs}
        else:
            pos_out = self.decoder.forward_position(net_out["feats"])
            tri_feats = self.triplane_processor(tri_net_out["tri_feats"], pos_out["xyz"])
            gs = self.decoder(tri_feats, partial_output=pos_out)
            outputs = {"pred_gs": gs}
        return outputs


class Gamba(torch.nn.Module):
    def __init__(self,
        opt: Options,
        **model_kwargs,
    ):
        super().__init__()
        self.model = GSPredictor(opt, **model_kwargs)
        self.opt = opt
        self.gs_render = GaussianRenderer(opt)
        if opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)

    def forward(self, data, step_ratio=1):
        results = {}
        loss = 0

        cond_poses = data['input']['camposes']
        cond_views = data['input']['images']
        plucker_img = data['input'].get('plucker_img', None)
        #flatten
        cond_poses = cond_poses.view(cond_poses.size(0), cond_poses.size(1), -1)  # (bsz, view_num, 16)
        

        decoder_out = self.model(cond_views=cond_views, # (bsz, view_num, c, h, w)
                         cam_poses=cond_poses, plucker_img=plucker_img) # (bsz, view_num, 16)
        
        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32, device=cond_views.device)
        # use the other views for rendering and supervision
        results = self.gs_render.render(decoder_out['pred_gs'], data['cam_view'], 
                                        data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        pred_images = results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas

        gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output'] # [B, V, 1, output_size, output_size], ground-truth masks

        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)
        loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
        loss = loss + loss_mse

        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                # gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # downsampled to at most 256 to reduce memory cost
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.opt.lambda_lpips * loss_lpips
            
        results['loss'] = loss

        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr

        return results


    def prepare_default_rays(self, device, elevation=0):
        from kiui.cam import orbit_camera
        from core.utils import get_rays
        import numpy as np

        cam_poses = np.stack([
            orbit_camera(elevation, 0, radius=self.opt.cam_radius),
            # orbit_camera(elevation, 90, radius=self.opt.cam_radius),
            # orbit_camera(elevation, 180, radius=self.opt.cam_radius),
            # orbit_camera(elevation, 270, radius=self.opt.cam_radius),
        ], axis=0) # [4, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

            ## visualize rays for plotting figure
            # kiui.vis.plot_image(rays_d * 0.5 + 0.5, save=True)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
        
        return rays_embeddings

    @torch.no_grad
    def forward_gaussians(self, data):
        # cond_poses = data['input']['camposes']
        cond_views = data['images']
        plucker_img = data['plucker_img']
        # cond_poses = cond_poses.view(cond_poses.size(0), cond_poses.size(1), -1)  # (bsz, view_num, 16)
        decoder_out = self.model(cond_views=cond_views, # (bsz, view_num, c, h, w)
                         cam_poses=None, plucker_img=plucker_img) # (bsz, view_num, 16)
        
        return decoder_out["pred_gs"]