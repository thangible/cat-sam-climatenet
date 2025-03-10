from train_parser import parse
import os
import random
import numpy as np
import torch
import wandb


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_with_projection(image, mask, prediction, label, var_names, use_projection=False, batch_num=None, epoch=None):
    # Convert tensors to numpy arrays
    # Check if the image tensor needs to be transposed
    if image.ndim == 3 and image.shape[0] in [1, 3]:
        image_np = image.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
    else:
        image_np = image.cpu().numpy()
    mask_np = mask.cpu().numpy().squeeze() if torch.is_tensor(mask) else mask.squeeze()  # Remove channel dimension
    prediction_np = prediction.detach().cpu().numpy().squeeze() if torch.is_tensor(prediction) else prediction.squeeze()  # Remove channel dimension

    longitudes = np.linspace(-180, 180, image_np.shape[1])
    latitudes = np.linspace(-90, 90, image_np.shape[0])

    # Normalize image data to [0, 1] range for imshow
    image_np = image_np / 255.0

    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()} if use_projection else {})

    # Plot the RGB image
    ax.imshow(image_np, origin='upper', extent=[-180, 180, -90, 90] if use_projection else None, alpha = 0.7)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black')

    # Plot the mask and prediction contours
    if mask_np.ndim == 3:
        for i in range(mask_np.shape[0]):
            ax.contour(longitudes, latitudes, mask_np[i], colors='green', linewidths=1, levels=[0.5], transform=ccrs.PlateCarree() if use_projection else None)
    else:
        ax.contour(longitudes, latitudes, mask_np, colors='green', linewidths=1, levels=[0.5], transform=ccrs.PlateCarree() if use_projection else None)

    if prediction_np.ndim == 3:
        for i in range(prediction_np.shape[0]):
            ax.contour(longitudes, latitudes, prediction_np[i], colors='red', linewidths=1, levels=[0.5], transform=ccrs.PlateCarree() if use_projection else None)
    else:
        ax.contour(longitudes, latitudes, prediction_np, colors='red', linewidths=1, levels=[0.5], transform=ccrs.PlateCarree() if use_projection else None)

    # Add a legend
    red_path = plt.Line2D([0], [0], color='red', linewidth=1, label='Prediction')
    green_path = plt.Line2D([0], [0], color='green', linewidth=1, label='Ground Truth')
    plt.legend(handles=[red_path, green_path], loc='upper right')

    # Add title and labels
    title = f'World projection with RGB as {var_names[0]}, {var_names[1]}, {var_names[2]} - Epoch {epoch} - {label}'
    plt.title(title)

    # Save the plot to a file with epoch and batch number
    filename = f'{label}_epoch_{epoch}.png'
    plt.savefig(filename)
    plt.close(fig)

    # Log the image to wandb
    wandb.log({"Validation example": wandb.Image(filename, caption=title)})


def calculate_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    assert inputs.size(0) == targets.size(0)
    inputs = inputs.sigmoid()
    inputs, targets = inputs.flatten(1), targets.flatten(1)

    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


def worker_init_fn(worker_id: int, base_seed: int, same_worker_seed: bool = True):
    """
    Set random seed for each worker in DataLoader to ensure the reproducibility.

    """
    seed = base_seed if same_worker_seed else base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)



def batch_to_cuda(batch, device):
    for key in batch.keys():
        if key in ['images', 'gt_masks', 'point_coords', 'box_coords', 'noisy_object_masks', 'object_masks']:
            batch[key] = [
                item.to(device=device, dtype=torch.float32) if item is not None else None for item in batch[key]
            ]
        elif key in ['point_labels']:
            batch[key] = [
                item.to(device=device, dtype=torch.long) if item is not None else None for item in batch[key]
            ]
    return batch