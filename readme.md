
## Gamba

This is the official implementation of *Gamba: Marry Gaussian Splatting with Mamba for single view 3D reconstruction*.

### [Project Page](https://florinshen.github.io/gamba-project) | [Arxiv](https://arxiv.org/abs/2403.18795) | [Weights](https://huggingface.co/florinshen/Gamba)

<!-- https://github.com/florinshen/gamba-project/raw/webpage/video/gamba-teaser.mp4 -->

<video>
  <source src="./assets/gamba-teaser.mp4" type="video/mp4">
</video>

### Install

```bash
# xformers is required! please refer to https://github.com/facebookresearch/xformers for details.
# for example, we use torch 2.1.0 + cuda 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118

git clone --recursive git@github.com:SkyworkAI/Gamba.git
# a modified gaussian splatting (+ depth, alpha rendering)
pip install ./submodules/diff-gaussian-rasterization
# radial polygon mask, only in training,
pip install ./submodules/rad-polygon-mask

# for mesh extraction
pip install git+https://github.com/NVlabs/nvdiffrast

# other dependencies
pip install -r requirements.txt
```

### Pretrained Weights

Our pretrained weight can be downloaded from [huggingface](https://huggingface.co/ashawkey/LGM).

For example, to download the fp16 model for inference:
```bash
mkdir checkpoint && cd checkpoint
wget https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16.safetensors
cd ..
```

### Inference

Inference takes about 10GB GPU memory (loading all imagedream, mvdream, and our LGM).

```bash
### gradio app for both text/image to 3D
python app.py big --resume pretrained/model_fp16.safetensors

### test
# --workspace: folder to save output (*.ply and *.mp4)
# --test_path: path to a folder containing images, or a single image
python infer.py big --resume pretrained/model_fp16.safetensors --workspace workspace_test --test_path data_test 

### local gui to visualize saved ply
python gui.py big --output_size 800 --test_path workspace_test/saved.ply

### mesh conversion
python convert.py big --test_path workspace_test/saved.ply
```

For more options, please check [options](./core/options.py).

### Training

**NOTE**: 
Since the dataset used in our training is based on AWS, it cannot be directly used for training in a new environment.
We provide the necessary training code framework, please check and modify the [dataset](./core/provider_objaverse.py) implementation!

We also provide the **~80K subset of [Objaverse](https://objaverse.allenai.org/objaverse-1.0)** used to train LGM in [objaverse_filter](https://github.com/ashawkey/objaverse_filter).

```bash
# debug training
accelerate launch --config_file acc_configs/gpu1.yaml main.py big --workspace workspace_debug

# training (use slurm for multi-nodes training)
accelerate launch --config_file acc_configs/gpu8.yaml main.py small --workspace /root/Results/workspace_plant



# training (gamba)
accelerate launch --config_file acc_configs/gpu8.yaml main.py gamba --workspace /mnt/xuanyuyi/results/gamba_human

```
### Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [LGM](https://github.com/3DTopia/LGM)
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [dearpygui](https://github.com/hoffstadt/DearPyGui)
- [tyro](https://github.com/brentyi/tyro)

### Citation

```
@article{shen2024gamba,
  title={Gamba: Marry gaussian splatting with mamba for single view 3d reconstruction},
  author={Shen, Qiuhong and Wu, Zike and Yi, Xuanyu and Zhou, Pan and Zhang, Hanwang and Yan, Shuicheng and Wang, Xinchao},
  journal={arXiv preprint arXiv:2403.18795},
  year={2024}
}
```
