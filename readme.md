
## Gamba

This is the official implementation of *Gamba: Marry Gaussian Splatting with Mamba for single view 3D reconstruction*.

### [Project Page](https://florinshen.github.io/gamba-project) | [Arxiv](https://arxiv.org/abs/2403.18795) | [Weights](https://huggingface.co/florinshen/Gamba)

### Why Gamba
🔥 Reconstruct 3D object from a single image input within 50 milliseconds. 


🔥 First end-to-end trainable single-view reconstruction model with 3DGS.

https://github.com/SkyworkAI/Gamba/assets/44775545/21bdc4e7-e070-446a-8fb7-401c9ee69921

### Install

```bash
# xformers is required! please refer to https://github.com/facebookresearch/xformers for details.
# for example, we use torch 2.1.0 + cuda 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install causal-conv1d==1.2.0 mamba-ssm
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

Our pretrained weight can be downloaded from [huggingface](https://huggingface.co/florinshen/Gamba). A lager Model is comming on the way!

For example, to download the bf16 model for inference:
```bash
mkdir checkpoint && cd checkpoint
wget https://huggingface.co/florinshen/Gamba/resolve/main/gamba_ep399.pth
cd ..
```

### Inference

Inference takes about 1.5GB GPU memory within 50 milliseconds.

```bash
bash scripts/test.sh
```

For more options, please check [options](./core/options.py).

### Training

We will update training tutorials soon. 


### Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [LGM](https://github.com/3DTopia/LGM)
- [OpenLRM](https://github.com/3DTopia/OpenLRM)
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [dearpygui](https://github.com/hoffstadt/DearPyGui)
- [tyro](https://github.com/brentyi/tyro)

### Citation

```bibtex
@article{shen2024gamba,
  title={Gamba: Marry gaussian splatting with mamba for single view 3d reconstruction},
  author={Shen, Qiuhong and Wu, Zike and Yi, Xuanyu and Zhou, Pan and Zhang, Hanwang and Yan, Shuicheng and Wang, Xinchao},
  journal={arXiv preprint arXiv:2403.18795},
  year={2024}
}
```
```bibtex
@article{yi2024mvgamba,
  title={MVGamba: Unify 3D Content Generation as State Space Sequence Modeling},
  author={Yi, Xuanyu and Wu, Zike and Shen, Qiuhong and Xu, Qingshan and Zhou, Pan and Lim, Joo-Hwee and Yan, Shuicheng and Wang, Xinchao and Zhang, Hanwang},
  journal={arXiv preprint arXiv:2406.06367},
  year={2024}
}
```