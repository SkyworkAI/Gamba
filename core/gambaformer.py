import math 

import torch
import torch.nn as nn

from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from timm.models.layers import DropPath, to_2tuple
# from gsutils.typings import *

# Basic types
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from torch import Tensor
import pdb

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        
        # drop path 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    
def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class ModLN(nn.Module):
    """
    Modulation with adaLN.
    
    References:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L101
    """
    def __init__(self, inner_dim: int, mod_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mod_dim, inner_dim * 2),
        )

    @staticmethod
    def modulate(x, shift, scale):
        # x: [N, L, D]
        # shift, scale: [N, D]
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x, cond):
        shift, scale = self.mlp(cond).chunk(2, dim=-1)  # [N, D]
        return self.modulate(self.norm(x), shift, scale)  # [N, L, D]


class ConditionModulationBlock(nn.Module):
    """
    Transformer block that takes in a cross-attention condition and another modulation vector applied to sub-blocks.
    """
    def __init__(self, inner_dim: int, cond_dim: int, mod_dim: int,
                 drop_path_rate: float = 0., layer_idx=0,
                 residual_in_fp32=False,
                 rms_norm=False,
                 fused_add_norm=False,
                 token_pnum=1,
                 token_num=-1):
        super().__init__()
        # self.camera_affine = nn.Linear(mod_dim, inner_dim)
        self.cond_affine = nn.Linear(cond_dim, inner_dim)
        
        self.mamba_block = create_block(d_model=inner_dim, 
                                        ssm_cfg=None,
                                        norm_epsilon=1e-5,
                                        rms_norm=rms_norm,
                                        residual_in_fp32=residual_in_fp32,
                                        fused_add_norm=fused_add_norm,
                                        layer_idx=layer_idx,
                                        drop_path=drop_path_rate,
                                        )
        self.token_pnum = token_pnum # token partition number
        self.token_num = token_num 

    def forward(self, hidden_states, residual, cond, mod, inference_params=None):
        cond = self.cond_affine(cond) # (bsz, 1024, inner_dim)
        # mod = self.camera_affine(mod)[:, None] #  -> (bsz, 1, inner_dim)
        # prepend_feats = torch.cat([mod, cond], dim=1)
        prepend_feats = cond
        COND_LEN = prepend_feats.size(1)
        TOKEN_LEN = self.token_num
        assert TOKEN_LEN % self.token_pnum == 0, f"error token number {TOKEN_LEN}."
        TOKEN_PLEN = TOKEN_LEN // self.token_pnum
        PART_LEN = COND_LEN + TOKEN_PLEN
        if residual is None:
            hidden_list = []
            for idx in range(self.token_pnum):
                hidden_list.extend([prepend_feats, hidden_states[:, idx * TOKEN_PLEN : (idx + 1) * TOKEN_PLEN]])
            hidden_states = torch.cat(hidden_list, dim=1).contiguous()
        else:
            hidden_list, residual_list = [], []
            for idx in range(self.token_pnum):
                hidden_list.append(hidden_states[:, idx * PART_LEN : (idx + 1) * PART_LEN][:, COND_LEN:])
                residual_list.append(residual[:, idx * PART_LEN : (idx + 1) * PART_LEN][:, COND_LEN:])
            hidden_states = torch.cat(hidden_list, dim=1).contiguous()
            residual = torch.cat(residual_list, dim=1).contiguous()
            hidden_list, residual_list = [], []
            for idx in range(self.token_pnum):
                hidden_list.extend([prepend_feats, hidden_states[:, idx * TOKEN_PLEN : (idx + 1) * TOKEN_PLEN]])
                residual_list.extend([torch.zeros_like(prepend_feats), residual[:, idx * TOKEN_PLEN : (idx + 1) * TOKEN_PLEN]])
            hidden_states = torch.cat(hidden_list, dim=1).contiguous()
            residual = torch.cat(residual_list, dim=1).contiguous()
        hidden_states, residual = self.mamba_block(hidden_states, residual, inference_params)
        return hidden_states, residual

    def set_token_num(self, token_num):
        if self.token_num > 0:
            return 
        else:
            self.token_num = token_num 
            return 
        
class GambaFormer(nn.Module):
    def __init__(self, 
                 inner_dim: int, image_feat_dim: int, 
                 mod_embed_dim: int, num_layers: int, 
                 gs_num:int, 
                 token_pnum: int = 1,
                 drop_path_rate: float = 0.1,
                 fused_add_norm=True, # False
                 rms_norm=True,
                 norm_epsilon=1e-5,
                 residual_in_fp32=True,
                 initializer_cfg=None):
        super().__init__()        
        self.gs_num = gs_num
        self.token_pnum = token_pnum
        self.pos_embed = nn.Parameter(torch.randn(gs_num, inner_dim) * (1. / inner_dim) ** 0.5)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.layers = nn.ModuleList([
            ConditionModulationBlock(
                inner_dim=inner_dim, cond_dim=image_feat_dim, 
                mod_dim=mod_embed_dim, drop_path_rate=inter_dpr[i],
                layer_idx=i,
                token_pnum=token_pnum,
                token_num=gs_num,)
            for i in range(num_layers)
        ])

        factory_kwargs = {"device": None, "dtype": None}
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            inner_dim, eps=norm_epsilon, **factory_kwargs
        )

        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32

        self.apply(
            partial(
                _init_weights,
                n_layer=num_layers,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
    
    def forward(self, img_cond, mod, inference_params=None, plucker_cond=None):
        N, L, _ = img_cond.shape
        gs_tokens = self.pos_embed.repeat(N, 1, 1)
        if plucker_cond is not None:
            gs_tokens = gs_tokens + plucker_cond
        hidden_states, residual = gs_tokens, None
        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(hidden_states, residual, img_cond, mod, inference_params)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )    

        # COND_LEN = L + 1
        COND_LEN = L
        TOKEN_LEN = self.gs_num
        assert TOKEN_LEN % self.token_pnum == 0, f"error token number {TOKEN_LEN}."
        TOKEN_PLEN = TOKEN_LEN // self.token_pnum
        PART_LEN = COND_LEN + TOKEN_PLEN

        hidden_list = []
        for idx in range(self.token_pnum):
            hidden_list.append(hidden_states[:, idx * PART_LEN : (idx + 1) * PART_LEN][:, COND_LEN:])
        hidden_states = torch.cat(hidden_list, dim=1).contiguous()
        feats = hidden_states
        return {"feats": feats}


class TriGambaFormer(nn.Module):
    def __init__(self, 
                 inner_dim: int, image_feat_dim: int, 
                 mod_embed_dim: int, num_layers: int, 
                 drop_path_rate: float = 0.1,
                 fused_add_norm=True, # False
                 rms_norm=True,
                 norm_epsilon=1e-5,
                 residual_in_fp32=True,
                 initializer_cfg=None):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.layers = nn.ModuleList([
            ConditionModulationBlock(
                inner_dim=inner_dim, cond_dim=image_feat_dim, 
                mod_dim=mod_embed_dim, drop_path_rate=inter_dpr[i],
                layer_idx=i,)
            for i in range(num_layers)
        ])

        factory_kwargs = {"device": None, "dtype": None}
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            inner_dim, eps=norm_epsilon, **factory_kwargs
        )

        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32

        self.apply(
            partial(
                _init_weights,
                n_layer=num_layers,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
    
    def forward(self, img_cond, mod, inference_params=None, embedding=None):
        assert embedding is not None
        N, L, _ = img_cond.shape
        token_num = embedding.shape[0]
        hidden_states, residual = embedding.repeat(N, 1, 1), None
        for idx, layer in enumerate(self.layers):
            layer.set_token_num(token_num)
            hidden_states, residual = layer(hidden_states, residual, img_cond, mod, inference_params)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )    
    
        feats = hidden_states[:, L + 1:, :]
        return {"tri_feats": feats}

        
if __name__ == "__main__":
    model = GambaFormer(inner_dim=512, 
                  image_feat_dim=768, 
                  mod_embed_dim=128, 
                  num_layers=8, 
                  gs_num=16384, 
                  drop_path_rate=0.1).cuda().train()
    import pdb 
    img_cond = torch.randn(1, 1024, 768).cuda()
    mod = torch.randn(1, 128).cuda()
    pdb.set_trace()
    output = model(img_cond, mod)
    pdb.set_trace()
    print(output)
