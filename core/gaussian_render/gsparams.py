import torch
from torch import nn



class GaussianModel(nn.Module):
    def __init__(self, params, color=None, SH_degree=0):
        """
        predicted 3dgs parameters format (num_pts, 14)
        xyz (3) + scale (3) + rot(4) + opacity(1) + SH_0 (3)
        """
        super(GaussianModel, self).__init__()
        # assert params.size(-1) == (11 + 3 * (SH_degree + 1) ** 2), ""
        assert params.size(-1) == (11 + 3), ""

        self.xyz = params[:, :3]
        self.scale = params[:, 3:6]
        self.rotation = params[:, 6:10]
        self.opacity = params[:, 10:11]
        self.SHs = params[:, 11:]
    
        self.active_sh_degree = SH_degree

        # set background as black, as rendered image is black bg
        if color is None:
            self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        else:
            self.background = color
    @property
    def get_bg_color(self):
        return self.background
    
    @property 
    def get_xyz(self):
        return self.xyz
    
    @property
    def get_opacity(self):
        return self.opacity 
    
    @property 
    def get_scaling(self):
        return self.scale 

    @property 
    def get_rotation(self):
        return self.rotation 
    
    @property 
    def get_features(self):
        # return self.SHs.reshape(self.SHs.size(0), -1, 3).contiguous()
        return self.SHs # (B, 3)
    

class BatchGaussians(list):
    def __init__(self, gsparams, bg_color=None, SH_degree=0):
        super(BatchGaussians, self).__init__()
        if bg_color is None:
            for param in gsparams:
                self.append(GaussianModel(param, SH_degree=SH_degree))
        else:
            for param, color in zip(gsparams, bg_color):
                self.append(GaussianModel(param, color, SH_degree=SH_degree))
    
    def index(self, *args, **kwargs) -> GaussianModel:
        super().index(*args, **kwargs)