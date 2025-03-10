import argparse
import os
import glob
import random
import wandb
from contextlib import nullcontext
from functools import partial
from os.path import join
from torchvision.utils import make_grid

import torch.nn.functional as F
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from torch.utils.data import DataLoader
from tqdm import tqdm
from misc import get_idle_gpu, get_idle_port, set_randomness

from cat_sam.datasets.whu import WHUDataset
from cat_sam.datasets.kvasir import KvasirDataset
from cat_sam.datasets.sbu import SBUDataset
from cat_sam.datasets.climatenet import ClimateDataset
from cat_sam.datasets.transforms import HorizontalFlip, VerticalFlip, RandomCrop
from cat_sam.models.modeling import CATSAMT, CATSAMA
from cat_sam.utils.evaluators import SamHQIoU, StreamSegMetrics

from train_util import parse, batch_to_cuda, calculate_dice_loss, plot_with_projection, worker_init_fn



wandb.init(project="cat-sam-climatenet", config={

})


def initialize_worker(worker_id, worker_args):
    set_randomness()
    if isinstance(worker_id, str):
        worker_id = int(worker_id)
    return worker_id

def setup_device_and_distributed(worker_id, worker_args):
    gpu_num = len(worker_args.used_gpu)
    world_size = os.environ['WORLD_SIZE'] if 'WORLD_SIZE' in os.environ.keys() else gpu_num
    base_rank = os.environ['RANK'] if 'RANK' in os.environ.keys() else 0
    local_rank = base_rank * gpu_num + worker_id
    if gpu_num > 1:
        dist.init_process_group(backend='nccl', init_method=worker_args.dist_url,
                                world_size=world_size, rank=local_rank)
    device = torch.device(f"cuda:{worker_id}")
    torch.cuda.set_device(device)
    return device, local_rank

def prepare_datasets(worker_args):
    if worker_args.cat_type == 'cat-t' and worker_args.dataset in ['kvasir', 'sbu']:
        transforms = [VerticalFlip(p=0.5), HorizontalFlip(p=0.5)]
    else:
        transforms = [VerticalFlip(p=0.5), HorizontalFlip(p=0.5)] # , RandomCrop(scale=[0.1, 1.0], p=1.0)

    max_object_num = None
    if worker_args.dataset == 'whu':
        dataset_class = WHUDataset
        max_object_num = 25
    elif worker_args.dataset == 'kvasir':
        dataset_class = KvasirDataset
    elif worker_args.dataset == 'sbu':
        dataset_class = SBUDataset
    elif worker_args.dataset == 'climate':
        dataset_class = ClimateDataset
    else:
        raise ValueError(f'invalid dataset name: {worker_args.dataset}!')

    dataset_dir = join(worker_args.data_dir, worker_args.dataset)
    train_dataset = dataset_class(
        data_dir=dataset_dir, train_flag=True, shot_num=worker_args.shot_num,
        transforms=transforms, max_object_num=max_object_num
    )
    val_dataset = dataset_class(data_dir=dataset_dir, train_flag=False)
    return train_dataset, val_dataset

def create_dataloaders(train_dataset, val_dataset, worker_args):
    train_bs = worker_args.train_bs if worker_args.train_bs else (1 if worker_args.shot_num == 1 else 4)
    val_bs = worker_args.val_bs if worker_args.val_bs else 2
    train_workers, val_workers = 1 if worker_args.shot_num == 1 else 4, 2
    if worker_args.num_workers is not None:
        train_workers, val_workers = worker_args.num_workers, worker_args.num_workers

    sampler = None
    if torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_bs = int(train_bs / torch.distributed.get_world_size())
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=train_bs, shuffle=sampler is None, num_workers=train_workers,
        sampler=sampler, drop_last=False, collate_fn=train_dataset.collate_fn,
        worker_init_fn=partial(worker_init_fn, base_seed=3407)
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=val_bs, shuffle=False, num_workers=val_workers,
        drop_last=False, collate_fn=val_dataset.collate_fn
    )
    return train_dataloader, val_dataloader

def initialize_model(worker_args, device, local_rank):
    if worker_args.cat_type == 'cat-t':
        model_class = CATSAMT
    elif worker_args.cat_type == 'cat-a':
        model_class = CATSAMA
    else:
        raise ValueError(f'invalid cat_type: {worker_args.cat_type}!')
    model = model_class(model_type=worker_args.sam_type).to(device=device)
    if torch.distributed.is_initialized():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )
    return model

def setup_optimizer_and_scheduler(model, worker_args):
    lr = worker_args.lr if hasattr(worker_args, 'lr') else 1e-3
    weight_decay = worker_args.weight_decay if hasattr(worker_args, 'weight_decay') else 1e-4
    optimizer = torch.optim.AdamW(
        params=[p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )
    if worker_args.shot_num is None:
        max_epoch_num, valid_per_epochs = 50, 1
    elif worker_args.shot_num == 1:
        max_epoch_num, valid_per_epochs = 2000, 20
    elif worker_args.shot_num == 16:
        max_epoch_num, valid_per_epochs = 200, 2
    else:
        raise RuntimeError("Invalid shot number provided. Expected values are None, 1, or 16.")
    
    if worker_args.max_epoch_num:
        max_epoch_num = worker_args.max_epoch_num
        
    if worker_args.valid_per_epochs:
        valid_per_epochs = worker_args.valid_per_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=max_epoch_num, eta_min=1e-5
    )
    return optimizer, scheduler, max_epoch_num, valid_per_epochs




def initialize_evaluator(worker_args, train_dataset):
    if worker_args.dataset == 'hqseg44k':
        return SamHQIoU()
    else:
        class_names = train_dataset.class_names if worker_args.dataset in ['jsrt', 'fls'] else ['Background', 'Foreground']
        return StreamSegMetrics(class_names=class_names)


def setup_experiment_path(worker_args):
    return join(
        worker_args.exp_dir,
        f'{worker_args.dataset}_{worker_args.sam_type}_{worker_args.cat_type}_{worker_args.shot_num if worker_args.shot_num else "full"}shot'
    )


def train_one_epoch(epoch, train_dataloader, model, optimizer, scheduler, device, local_rank, worker_args, max_epoch_num):
    if hasattr(train_dataloader.sampler, 'set_epoch'):
        train_dataloader.sampler.set_epoch(epoch)

    train_pbar = tqdm(total=len(train_dataloader), desc='train', leave=False) if local_rank == 0 else None
    for train_step, batch in enumerate(train_dataloader):
        batch = batch_to_cuda(batch, device)
        masks_pred = model(
            imgs=batch['images'], point_coords=batch['point_coords'], point_labels=batch['point_labels'],
            box_coords=batch['box_coords'], noisy_masks=batch['noisy_object_masks']
        )
        masks_gt = batch['object_mask']
        masks_pred, masks_gt = preprocess_masks(masks_pred, masks_gt)

        total_loss, loss_dict = calculate_losses(masks_pred, masks_gt)
        log_training_metrics(epoch, train_step, masks_pred, masks_gt, loss_dict)

        backward_context = model.no_sync if torch.distributed.is_initialized() else nullcontext
        with backward_context():
            total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        reduce_losses(loss_dict)

        if train_pbar:
            update_progress_bar(train_pbar, epoch, max_epoch_num, loss_dict)

    scheduler.step()
    if train_pbar:
        train_pbar.clear()


def preprocess_masks(masks_pred, masks_gt):
    for masks in [masks_pred, masks_gt]:
        for i in range(len(masks)):
            if len(masks[i].shape) == 2:
                masks[i] = masks[i][None, None, :]
            if len(masks[i].shape) == 3:
                masks[i] = masks[i][:, None, :]
            if len(masks[i].shape) != 4:
                raise RuntimeError
    return masks_pred, masks_gt


def calculate_losses(masks_pred, masks_gt):
    bce_loss_list, dice_loss_list = [], []
    for i in range(len(masks_pred)):
        pred, label = masks_pred[i], masks_gt[i]
        label = torch.where(torch.gt(label, 0.), 1., 0.)
        b_loss = F.binary_cross_entropy_with_logits(pred, label.float())
        d_loss = calculate_dice_loss(pred, label)

        bce_loss_list.append(b_loss)
        dice_loss_list.append(d_loss)

    bce_loss = sum(bce_loss_list) / len(bce_loss_list)
    dice_loss = sum(dice_loss_list) / len(dice_loss_list)
    total_loss = bce_loss + dice_loss
    loss_dict = dict(
        total_loss=total_loss.clone().detach(),
        bce_loss=bce_loss.clone().detach(),
        dice_loss=dice_loss.clone().detach()
    )
    return total_loss, loss_dict


def log_training_metrics(epoch, train_step, masks_pred, masks_gt, loss_dict):
    with torch.no_grad():
        pred_labels = (torch.sigmoid(masks_pred[0]) > 0.5).float()
        true_labels = masks_gt[0]
        intersection = (pred_labels * true_labels).sum()
        union = pred_labels.sum() + true_labels.sum() - intersection
        iou = intersection / union if union != 0 else torch.tensor(0.0)
        precision = intersection / pred_labels.sum() if pred_labels.sum() != 0 else torch.tensor(0.0)
        recall = intersection / true_labels.sum() if true_labels.sum() != 0 else torch.tensor(0.0)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else torch.tensor(0.0)

    wandb.log({
        "epoch": epoch,
        "train_step": train_step,
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1_score": f1_score.item(),
        "total_loss": loss_dict['total_loss'].item(),
        "bce_loss": loss_dict['bce_loss'].item(),
        "dice_loss": loss_dict['dice_loss'].item()
    })


def reduce_losses(loss_dict):
    if torch.distributed.is_initialized():
        for key in loss_dict.keys():
            if hasattr(loss_dict[key], 'detach'):
                loss_dict[key] = loss_dict[key].detach()
            torch.distributed.reduce(loss_dict[key], dst=0, op=torch.distributed.ReduceOp.SUM)
            loss_dict[key] /= torch.distributed.get_world_size()


def update_progress_bar(train_pbar, epoch, max_epoch_num, loss_dict):
    train_pbar.update(1)
    str_step_info = "Epoch: {epoch}/{epochs:4}. " \
                    "Loss: {total_loss:.4f}(total), {bce_loss:.4f}(bce), {dice_loss:.4f}(dice)".format(
        epoch=epoch, epochs=max_epoch_num,
        total_loss=loss_dict['total_loss'], bce_loss=loss_dict['bce_loss'], dice_loss=loss_dict['dice_loss']
    )
    train_pbar.set_postfix_str(str_step_info)


def validate_one_epoch(epoch, val_dataloader, model, iou_eval, device, exp_path, best_miou, worker_args, max_epoch_num):
    model.eval()
    valid_pbar = tqdm(total=len(val_dataloader), desc='valid', leave=False)
    for val_step, batch in enumerate(val_dataloader):
        batch = batch_to_cuda(batch, device)
        val_model = model.module if hasattr(model, 'module') else model

        with torch.no_grad():
            val_model.set_infer_img(img=batch['images'])
            masks_pred = val_model.infer(point_coords=batch['point_coords']) if worker_args.dataset == 'm_roads' else val_model.infer(box_coords=batch['box_coords'])

        masks_gt = batch['gt_masks']
        masks_pred, masks_gt = preprocess_masks(masks_pred, masks_gt)

        iou_eval.update(masks_gt, masks_pred, batch['index_name'])
        valid_pbar.update(1)
        str_step_info = "Epoch: {epoch}/{epochs:4}.".format(epoch=epoch, epochs=max_epoch_num)
        valid_pbar.set_postfix_str(str_step_info)

        metrics = iou_eval.compute()
        mean_iou = metrics[0]['Mean Foreground IoU']
        mean_precision = metrics[0]['Mean Foreground Precision']
        mean_recall = metrics[0]['Mean Foreground Recall']
        mean_f1 = metrics[0]['Mean Foreground F1']
        iou_eval.reset()
        valid_pbar.clear()

        wandb.log({
            "epoch": epoch,
            "val_step": val_step,
            "mean_iou": mean_iou,
            "mean_precision": mean_precision,
            "mean_recall": mean_recall,
            "mean_f1": mean_f1
        })

        if mean_iou > best_miou:
            
            if worker_args.save_model:
                torch.save(
                    model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
                    join(exp_path, "best_model.pth")
                )
            best_miou = mean_iou
            print(f'Best mIoU has been updated to {best_miou:.2%}!')
            wandb.save(join(exp_path, "best_model.pth"))

        log_images_to_wandb(batch, masks_pred, epoch, val_step)


def log_images_to_wandb(batch, masks_pred, epoch, train_step):
    wandb.log({
        "Images type": str(type(batch['images'])),
        "Images shape": [img.shape for img in batch['images'][:4]],
        "Masks type": str(type(batch['gt_masks'])),
        "Masks shape": [mask.shape for mask in batch['gt_masks'][:4]],
        "Preds type": str(type(masks_pred)),
        "Preds shape": [pred.shape for pred in masks_pred[:4]]
    })

    images = [img for img in batch['images'][:4]]
    masks = [mask for mask in batch['gt_masks'][:4]]
    preds = [pred for pred in masks_pred[:4]]

    for i in range(len(images)):
        plot_with_projection(images[i], masks[i], preds[i], use_projection=True, batch_num=train_step, epoch=epoch)
        
        
def main_worker(worker_id, worker_args):
    worker_id = initialize_worker(worker_id, worker_args)
    device, local_rank = setup_device_and_distributed(worker_id, worker_args)
    train_dataset, val_dataset = prepare_datasets(worker_args)
    train_dataloader, val_dataloader = create_dataloaders(train_dataset, val_dataset, worker_args)
    model = initialize_model(worker_args, device, local_rank)
    optimizer, scheduler, max_epoch_num, valid_per_epochs = setup_optimizer_and_scheduler(model, worker_args)

    best_miou = 0
    iou_eval = initialize_evaluator(worker_args, train_dataset)

    exp_path = setup_experiment_path(worker_args)
    os.makedirs(exp_path, exist_ok=True)

    for epoch in range(1, max_epoch_num + 1):
        train_one_epoch(epoch, train_dataloader, model, optimizer, scheduler, device, local_rank, worker_args, max_epoch_num)
        if local_rank == 0 and epoch % valid_per_epochs == 0:
            validate_one_epoch(epoch, val_dataloader, model, iou_eval, device, exp_path, best_miou, worker_args, max_epoch_num)

if __name__ == '__main__':
    args = parse()

    if torch.cuda.is_available():
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            used_gpu = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        else:
            used_gpu = get_idle_gpu(gpu_num=1)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(used_gpu[0])
        args.used_gpu, args.gpu_num = used_gpu, len(used_gpu)
    else:
        args.used_gpu, args.gpu_num = [], 0

    # launch the experiment process for both single-GPU and multi-GPU settings
    if len(args.used_gpu) == 1:
        main_worker(worker_id=0, worker_args=args)
    else:
        # initialize multiprocessing start method
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            try:
                mp.set_start_method('forkserver')
                print("Fail to initialize multiprocessing module by spawn method. "
                      "Use forkserver method instead. Please be careful about it.")
            except RuntimeError as e:
                raise RuntimeError(
                    "Your server supports neither spawn or forkserver method as multiprocessing start methods. "
                    f"The error details are: {e}"
                )

        # dist_url is fixed to localhost here, so only single-node DDP is supported now.
        args.dist_url = "tcp://127.0.0.1" + f':{get_idle_port()}'
        # spawn one subprocess for each GPU
        mp.spawn(main_worker, nprocs=args.gpu_num, args=(args,))
