import os
import cv2
import json
import random
import numpy as np
import pandas as pd

from os import path as osp
import tarfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import ElasticTransform
from torch.utils.data import Dataset

import kiui
import roma
from kiui.op import safe_normalize
from core.options import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
import pdb
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


# check camera settings, render size

def check_tar_integrity(tar_path):
    try:
        with tarfile.open(tar_path, 'r') as tar:
            tar.getmembers()  # Attempt to read all members of the tar file
        return True  # No error was raised, file is likely fine
    except tarfile.TarError as e:
        print(f"{tar_path}, Integrity check failed: {e}")
        return False  # An error was caught indicating corruption



class ObjaverseDataset(Dataset):
    def __init__(self, opt: Options, training=True):
        
        self.opt = opt
        self.training = training
        
        self.plucker_ray = opt.plucker_ray
        self.use_dino = opt.use_dino

        self.data_root = '/path/to/gobjaverse/'
        
        self.items = []

        with open(osp.join(self.data_root, 'gobj_lvis.json'), 'r') as f:
            self.items = json.load(f)

        # TODO: naive splitting
        if self.opt.overfit:  
            initial_batch = self.items[:self.opt.batch_size]
            if len(initial_batch) > 0:  
                num_repeats = len(self.items) // len(initial_batch)  
                self.items = (initial_batch * num_repeats)[:len(self.items)]
        elif self.training:
            self.items = self.items[:-self.opt.batch_size]
        else:
            self.items = self.items[-self.opt.batch_size:]
        # resolution mode (will randomly change during training)
        # 0: default render_size, will render normal and calc eikonal loss
        # 1: render_size * 2, no normal to allow larger resolution...
        self.mode = 1
        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1

    def __len__(self):
        return len(self.items)

    # this will be called prior to __getitem__ for batched sampler, where I can randomize the mode batch-wisely!
    def __getitems__(self, indices):

        if self.training:
            self.mode = random.randint(0, 1)
        else:
            self.mode = 1
            
        return [self.__getitem__(i) for i in indices]

    def __getitem__(self, idx):

        uid = self.items[idx]
        obj_valid = False 
        tar_path = os.path.join(self.data_root, 'savedata', f"{uid}.tar")
        while not obj_valid:
            if os.path.exists(tar_path) and check_tar_integrity(tar_path) :
                obj_valid = True
            else:
                idx = random.randint(0,len(self.items) - 1)
                uid = self.items[idx]
                tar_path = os.path.join(self.data_root, 'savedata', f"{uid}.tar")

        results = {}

        mode = self.mode
        results['mode'] = mode

        # load num_views images
        images = []
        masks = []
        depths = []
        normals = []
        cam_poses = []
        
        vid_cnt = 0

        if self.training:
            vids = [random.choice([0,6,12,18])] + np.random.choice(range(25), 12, replace=False).tolist() 
        else:
            vids = [0,6,12,18] + np.random.choice(range(25), 12, replace=False).tolist() 

        uid_last = uid.split('/')[1]
        tar_handler = tarfile.open(tar_path, 'r')

        for vid in vids:
            image_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}.png")
            meta_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}.json")
            # albedo_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}_albedo.png") # black bg...
            # mr_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}_mr.png")
            # nd_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}_nd.exr")
            
            # try:
            try:
                with tar_handler.extractfile(image_path) as f:
                    image = np.frombuffer(f.read(), np.uint8)
                with tar_handler.extractfile(meta_path) as f:
                    meta = json.loads(f.read().decode())
            except:
                # import pdb 
                # pdb.set_trace()
                continue
                
            image = torch.from_numpy(cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
            
            c2w = np.eye(4)
            c2w[:3, 0] = np.array(meta['x'])
            c2w[:3, 1] = np.array(meta['y'])
            c2w[:3, 2] = np.array(meta['z'])
            c2w[:3, 3] = np.array(meta['origin'])
            c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)
            
            # blender world + opencv cam --> opengl world & cam
            c2w[1] *= -1
            c2w[[1, 2]] = c2w[[2, 1]]
            c2w[:3, 1:3] *= -1 # invert up and forward direction

            # radius is random, normalize it... but this will lead to wrong depth scale, need to use scale-invariant depth loss
            dist = torch.norm(c2w[:3, 3]).item()
            c2w[:3, 3] *= self.opt.cam_radius / dist
          
            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]
            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb

            # normal = normal.permute(2, 0, 1) # [3, 512, 512]

            images.append(image)
            # normals.append(normal)
            # depths.append(depth)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break
        
        # close to avoid memory overflow
        tar_handler.close()

        if vid_cnt < self.opt.num_views:
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.opt.num_views - vid_cnt
            images = images + [images[-1]] * n
            # normals = normals + [normals[-1]] * n
            # depths = depths + [depths[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) # [V, C, H, W]
        # normals = torch.stack(normals, dim=0) # [V, C, H, W]
        # depths = torch.stack(depths, dim=0) # [V, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        ### inputs
        images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        dino_images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.dino_input_size, self.opt.dino_input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        # data augmentation
        # if self.training:
        #     # apply random grid distortion to simulate 3D inconsistency
        #     if random.random() < self.opt.prob_grid_distortion:
        #         images_input[1:] = grid_distortion(images_input[1:])
        #     # apply camera jittering (only to input!)
        #     if random.random() < self.opt.prob_cam_jitter:
        #         cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        if not self.use_dino:
            # if use orig_img, need to pre-process with mean and std.
            images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        if self.plucker_ray:
            # build ray embeddings for input views
            rays_embeddings = []
            for i in range(self.opt.num_input_views):
                rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
                rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
                rays_embeddings.append(rays_plucker)
            rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
            final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        else:
            final_input = {'images': dino_images_input, 'camposes': cam_poses_input}

            # also use the plucker image,
            images_input = TF.normalize(images_input.clone(), IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            rays_embeddings = []
            for i in range(self.opt.num_input_views):
                rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
                rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
                rays_embeddings.append(rays_plucker)
            rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
            plucker_img = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
            final_input.update({'plucker_img': plucker_img})

        results['input'] = final_input

        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]
        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]

        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos
        
        return results

if __name__ == "__main__":
    import torch
    import pdb
    import tyro
    from options import AllConfigs
    
    opt = tyro.cli(AllConfigs)
    train_dataset = ObjaverseDataset(opt=opt,training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    for data in train_dataloader:
        print(data.keys())
    