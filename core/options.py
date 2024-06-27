import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Dict, Optional


@dataclass
class Options:
    ### model
    # Unet image input size
    model_type: str = 'gamba' # or lgm
    plucker_ray: bool = False
    use_dino: bool = True
    overfit: bool = False

    input_size: int = 512 # be consistent to DINO, 336
    dino_input_size: int = 336
    num_input_views: int = 1 # set input view as 1
    dino_name: str = 'dinov2_vitb14_reg'
    dino_dim: int = 768
    
    patch_size: int = 8

    # model params
    gs_num: int = 16384
    token_pnum: int = 1 # partition tokens
    gamba_layers: int = 14
    gamba_dim: int = 512
    campose_dim: int = 128

    
    # model variants
    use_triplane: bool = False
    enable_triplane_epoch: int = 0
    triplane_dim: int = 80
    triplane_radius: float = 0.5 # 0.6 in tgs

    # Unet definition
    down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024, 1024)
    down_attention: Tuple[bool, ...] = (False, False, False, True, True, True)
    mid_attention: bool = True
    up_channels: Tuple[int, ...] = (1024, 1024, 512, 256)
    up_attention: Tuple[bool, ...] = (True, True, True, False)
    # Unet output size, dependent on the input_size and U-Net structure!
    splat_size: int = 64
    # gaussian render size
    output_size: int = 512 # output size

    ### dataset
    # data mode (only support s3 now)
    data_mode: Literal['s3'] = 's3'
    # fovy of the dataset
    fovy: float = 49.1 # 49.1
    # camera near plane
    znear: float = 0.5
    # camera far plane
    zfar: float = 2.5
    # number of all views (input + output)
    num_views: int = 3
    # number of views
    # num_input_views : int = 1
    num_output_views: int = 2
    # camera radius
    cam_radius: float = 1.5 # to better use [-1, 1]^3 space
    # num workers
    num_workers: int = 8

    ### training
    # workspace
    workspace: str = 'mnt/xuanyuyi/results/workspace'
    # resume
    resume: Optional[str] = None # a scan instead of convolution
    batch_size: int = 16
    # gradient accumulation
    gradient_accumulation_steps: int = 1
    # training epochs
    num_epochs: int = 400
    # lpips loss weight
    lambda_lpips: float = 0.5
    # gradient clip
    gradient_clip: float = 1.0
    # mixed precision
    mixed_precision: str = 'bf16' # bf16
    # learning rate
    lr: float = 2e-3
    # augmentation prob for grid distortion
    prob_grid_distortion: float = 0.5
    # augmentation prob for camera jitter
    prob_cam_jitter: float = 0.5
    warmup_epochs: int = 10

    ### testing
    # test image path
    test_path: Optional[str] = None

    ### misc
    # nvdiffrast backend setting
    force_cuda_rast: bool = False
    # render fancy video with gaussian scaling effect
    fancy_video: bool = False
    
    # renderig resolution zoom factor for patched rendering
    zoom: int = 3
# all the default settings
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['gamba'] = 'the default settings for Gamba'
config_defaults['gamba'] = Options()

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
