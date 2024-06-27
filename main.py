import tyro
import time
import random
import datetime
import torch
from core.options import AllConfigs
from core.gamba_models import Gamba
from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file
import os
import copy

import kiui
from core.utils import CosineWarmupScheduler
import wandb


def main():    
    opt = tyro.cli(AllConfigs)
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        # kwargs_handlers=[ddp_kwargs],
    )

    rebuild_model = False
    # model
    if opt.model_type == 'gamba':
        _opt = copy.deepcopy(opt)
        if opt.use_triplane and (opt.enable_triplane_epoch > 0):
            _opt.use_triplane = False
            rebuild_model = True
        model = Gamba(_opt)
    else:
        raise NotImplementedError

    # data
    if opt.data_mode == 's3':
        from core.provider_ikun import ObjaverseDataset as Dataset
    else:
        raise NotImplementedError
    
    train_dataset = Dataset(opt, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = Dataset(opt, training=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))

    # scheduler (per-iteration)
    total_steps = opt.num_epochs * int(len(train_dataloader) / opt.gradient_accumulation_steps)

    warmup_iters = opt.warmup_epochs * int(len(train_dataloader) / opt.gradient_accumulation_steps)
    scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup_iters=warmup_iters, max_iters=total_steps) 


    # resume
    start_epoch = 0
    legacy_load = False
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
            legacy_load = True
        elif opt.resume.endswith('pth'):
            ckpt = torch.load(opt.resume, map_location='cpu')
            if accelerator.is_main_process:
                print(f"load checkpoint from {opt.resume}")
            torch.distributed.barrier()
            if rebuild_model and (ckpt['epoch'] == opt.enable_triplane_epoch - 1):
                if accelerator.is_main_process:
                    print("enable triplane by rebuilding model")
                torch.distributed.barrier()
                model = Gamba(opt).train()
                missing_keys, unexpected_keys = model.load_state_dict(ckpt['model'], strict=False)
                optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))
                scheduler.load_state_dict(ckpt['scheduler'])
                new_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup_iters=warmup_iters, max_iters=total_steps) 
                for _ in range(scheduler._step_count):
                    new_scheduler.step()
                scheduler = new_scheduler
                rebuild_model = False 
                start_epoch = ckpt['epoch'] + 1
            legacy_load = False
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
            legacy_load = True
        
        # tolerant load (only load matching shapes)
        # model.load_state_dict(ckpt, strict=False)
        if legacy_load:
            state_dict = model.state_dict()
            for k, v in ckpt.items():
                if k in state_dict: 
                    if state_dict[k].shape == v.shape:
                        state_dict[k].copy_(v)
                    else:
                        accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
                else:
                    accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')
    
    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    if accelerator.is_main_process:
        wandb.login()
        wandb.init(
            project="single-gamba",
            name=opt.workspace.split("/")[-1],
            config=opt,
            dir=opt.workspace,
        )
        wandb.watch(model, log_freq=500)

    # loop
    start_time = datetime.datetime.now()
    for epoch in range(start_epoch, opt.num_epochs):
        if rebuild_model and (epoch >= opt.enable_triplane_epoch):
            if accelerator.is_main_process:
                print("enable triplane by rebuilding model")
            torch.distributed.barrier()
            # first save checkpoint 
            if accelerator.is_main_process:
                checkpoint = {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.optimizer.state_dict(),
                    'scheduler': scheduler.scheduler.state_dict(),
                    'epoch': epoch - 1
                }
                torch.save(checkpoint, os.path.join(opt.workspace, 'checkpoint_ep{:03d}.pth'.format(epoch - 1)))
            torch.distributed.barrier()
            new_model = Gamba(opt).train()
            missing_keys, unexpected_keys = new_model.load_state_dict(model.module.state_dict(), strict=False)
            model = new_model
            optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))
            new_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup_iters=warmup_iters, max_iters=total_steps) 
            for _ in range(scheduler.scheduler._step_count):
                new_scheduler.step()
            scheduler = new_scheduler
            model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, test_dataloader, scheduler
            )
            rebuild_model = False
        # train
        model.train()
        total_loss = 0
        total_psnr = 0
        total_loss_lpips = 0
        wandb_gt_image = None
        wandb_pred_image = None
        wandb_eval_gt_image = None
        wandb_eval_pred_image = None

        if epoch <= 5:
            train_dataloader.dataset.opt.num_views = 3
            test_dataloader.dataset.opt.num_views = 3
        elif (epoch > 5) and (epoch < 60):
            train_dataloader.dataset.opt.num_views = 5
            test_dataloader.dataset.opt.num_views = 5
        else:
            train_dataloader.dataset.opt.num_views = 7
            test_dataloader.dataset.opt.num_views = 7

        cur_iters = 0
        for i, data in enumerate(train_dataloader):
            cur_iters += 1
            with accelerator.accumulate(model):

                optimizer.zero_grad()
                if opt.overfit:
                    step_ratio = 0.0
                else:
                    step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs

                out = model(data, step_ratio)
                loss = out['loss']
                psnr = out['psnr']
                accelerator.backward(loss)

                # gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                optimizer.step()
                scheduler.step()

                total_loss += loss.detach()
                total_psnr += psnr.detach()
                if 'loss_lpips' in out:
                    total_loss_lpips += out['loss_lpips'].detach()

            if accelerator.is_main_process:
                # logging
                if i % 100 == 0:
                    mem_free, mem_total = torch.cuda.mem_get_info()  
                    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  
                    elapsed = datetime.datetime.now() - start_time
                    elapsed_str = str(elapsed).split('.')[0]  
                    print(f"[{current_time} INFO] {i}/{len(train_dataloader)} | "
                        f"Elapsed: {elapsed_str} | "
                        f"Mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G | "
                        f"LR: {scheduler.get_last_lr()[0]:.7f} | "
                        f"Step ratio: {step_ratio:.4f} | "
                        f"Loss: {loss.item():.6f}")
                
                # save log images
                if i % 200 == 0:
                    gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{opt.workspace}/train_gt_images_{epoch}_{i}.jpg', gt_images)

                    # gt_alphas = data['masks_output'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                    # gt_alphas = gt_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, gt_alphas.shape[1] * gt_alphas.shape[3], 1)
                    # kiui.write_image(f'{opt.workspace}/train_gt_alphas_{epoch}_{i}.jpg', gt_alphas)

                    pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/train_pred_images_{epoch}_{i}.jpg', pred_images)

                    wandb_gt_image = wandb.Image(gt_images, caption=f"train_gt_images_{epoch}_{i}")
                    wandb_pred_image = wandb.Image(pred_images, caption=f"train_pred_images_{epoch}_{i}")
                    # pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                    # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                    # kiui.write_image(f'{opt.workspace}/train_pred_alphas_{epoch}_{i}.jpg', pred_alphas)

        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        total_loss_lpips = accelerator.gather_for_metrics(total_loss_lpips).mean()
        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            total_loss_lpips /= len(train_dataloader)
            accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}")
            wandb.log({"Loss/train": total_loss, "PSNR/train": total_psnr, 
                       "Loss/loss_lpips": total_loss_lpips,
                       "LR/lr": scheduler.get_last_lr()[0]
                      }, step=epoch, commit=False)
            wandb.log({"train/gt_images": wandb_gt_image, "train/pred_images": wandb_pred_image}, step=epoch, commit=False)
            # save psnr file
            train_psnr_log_file = os.path.join(opt.workspace, "train_psnr_log.txt")
            with open(train_psnr_log_file, "a") as file:
                file.write(f"Epoch: {epoch}, PSNR: {total_psnr.item():.4f}\n")

        # checkpoint
        if epoch % 20 == 0 or epoch == opt.num_epochs - 1:
            accelerator.wait_for_everyone()
            accelerator.save_model(model, opt.workspace)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoint = {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.optimizer.state_dict(),
                    'scheduler': scheduler.scheduler.state_dict(),
                    'epoch': epoch
                }
                torch.save(checkpoint, os.path.join(opt.workspace, 'checkpoint_ep{:03d}.pth'.format(epoch)))
            accelerator.wait_for_everyone()

        if opt.overfit:
            # skip evaluation
            continue
        # eval
        with torch.no_grad():
            model.eval()
            total_psnr = 0
            for i, data in enumerate(test_dataloader):

                out = model(data)
    
                psnr = out['psnr']
                total_psnr += psnr.detach()
                
                # save some images
                if accelerator.is_main_process:
                    gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{opt.workspace}/eval_gt_images_{epoch}_{i}.jpg', gt_images)

                    pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/eval_pred_images_{epoch}_{i}.jpg', pred_images)

                    # pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                    # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                    # kiui.write_image(f'{opt.workspace}/eval_pred_alphas_{epoch}_{i}.jpg', pred_alphas)
                    wandb_eval_gt_image = wandb.Image(gt_images, caption=f"eval_gt_images_{epoch}_{i}")
                    wandb_eval_pred_image = wandb.Image(pred_images, caption=f"eval_pred_images_{epoch}_{i}")

            torch.cuda.empty_cache()

            total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
            if accelerator.is_main_process:
                wandb.log({"PSNR/eval": total_psnr}, step=epoch, commit=False)
                wandb.log({"eval/gt_images": wandb_eval_gt_image, "eval/pred_images": wandb_eval_pred_image}, step=epoch, commit=True)
                total_psnr /= len(test_dataloader)
                accelerator.print(f"[eval] epoch: {epoch} psnr: {psnr:.4f}")
                # save psnr file
                test_psnr_log_file = os.path.join(opt.workspace, "test_psnr_log.txt")
                with open(test_psnr_log_file, "a") as file:
                    file.write(f"Epoch: {epoch}, PSNR: {total_psnr.item():.4f}\n")



if __name__ == "__main__":
    main()
