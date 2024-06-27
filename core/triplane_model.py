import torch 
import torch.nn as nn 
import torch.nn.functional as F
from einops import rearrange

import math
from core.options import Options

from core.gsutils.typings import *

ValidScale = Union[Tuple[float, float], Num[Tensor, "2 D"]]

def scale_tensor(
    dat: Num[Tensor, "... D"], inp_scale: ValidScale, tgt_scale: ValidScale
):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    if isinstance(tgt_scale, Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


class TriPlaneModel(nn.Module):
    def __init__(self, 
                 opt: Options,
                 **kwargs):
        super().__init__()
        # (3, 32, 32)
        self.opt = opt
        self.plane_size = 32
        self.embeddings = nn.Parameter(
            torch.randn(
                (3, self.opt.gamba_dim, self.plane_size, self.plane_size))
            * 1.
            / math.sqrt(self.opt.gamba_dim)
        )

        # self.up_sampler = nn.ConvTranspose2d(self.opt.gamba_dim, 
        #                                      self.opt.triplane_dim, 
        #                                      kernel_size=2, stride=2)

        self.up_sampler = nn.Conv2d(self.opt.gamba_dim, 
                                             self.opt.triplane_dim, 
                                             kernel_size=1, stride=1)
        self.radius = opt.triplane_radius
        self.output_channels = int(3 * opt.triplane_dim)

    def query_triplane(
        self,
        triplanes,
        positions,
    ):
        batched = positions.ndim == 3
        if not batched:
            # no batch dimension
            triplanes = triplanes[None, ...]
            positions = positions[None, ...]

        positions = scale_tensor(positions, (-self.radius, self.radius), (-1, 1))
        indices2D = torch.stack(
                (positions[..., [0, 1]], positions[..., [0, 2]], positions[..., [1, 2]]),
                dim=-3,
            )
        out = F.grid_sample(
            rearrange(triplanes, "B Np Cp Hp Wp -> (B Np) Cp Hp Wp", Np=3),
            rearrange(indices2D, "B Np N Nd -> (B Np) () N Nd", Np=3),
            align_corners=False,
            mode="bilinear",
        )
        out = rearrange(out, "(B Np) Cp () N -> B N (Np Cp)", Np=3)
        if not batched:
            out = out.squeeze(0)

        return out

    
    def forward(self, fwd_embed, query_pts):
        # fwd_embed shape (bsz, 3, h, w, c) -> (bsz, 3, c, h, w)
        fwd_embed = fwd_embed.reshape(fwd_embed.size(0), 3, self.plane_size, self.plane_size, -1)
        fwd_embed = fwd_embed.permute(0, 1, 4, 2, 3).contiguous()
        triplanes_up = rearrange(
            self.up_sampler(
                rearrange(fwd_embed, "B Np Ci Hp Wp -> (B Np) Ci Hp Wp", Np=3)
            ),
            "(B Np) Co Hp Wp -> B Np Co Hp Wp",
            Np=3,
        )

        return self.query_triplane(triplanes_up, query_pts.detach())
    

    def get_embedding(self):
        # (3, c, h, w) -> (c, 3, h, w) -> (c, 3 * h * w) -> (3 * h * w, c)
        return self.embeddings.permute(1, 0, 2, 3).flatten(1).permute(1, 0).contiguous()
    