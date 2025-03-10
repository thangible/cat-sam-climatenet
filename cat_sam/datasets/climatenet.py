import os
import random
import xarray as xr
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from cat_sam.datasets.misc import generate_prompts_from_mask
from cat_sam.datasets.base import BinaryCATSAMDataset  
from cat_sam.datasets.transforms import Compose
import cv2

class ClimateDataset(Dataset):
    def __init__(self, data_dir, train_flag=True, transforms=None, **prompt_kwargs):
        """
        Parameters:
            data_dir (str): Directory containing the .nc files.
            train_flag (bool): Whether the dataset is used for training.
            transforms (list): A list of transforms to apply.
            prompt_kwargs: Additional keyword arguments for prompt generation.
        """
        train_path = os.path.join(data_dir, "train")
        test_path = os.path.join(data_dir, "test")
        sub_dir = train_path if train_flag else test_path
        # print(sub_dir)
        self.files = [os.path.join(sub_dir, f) for f in sorted(os.listdir(sub_dir)) if f.endswith(".nc")]
        if len(self.files) == 0:
            raise ValueError(f"No .nc files found in directory: {sub_dir}")
        # print(len(self.files))
        self.train_flag = train_flag
        self.transforms = Compose(transforms) if transforms else None
        
        # Store prompt generation parameters.
        
        self.prompt_kwargs = prompt_kwargs

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Use filename as the unique index name.
        file_path = self.files[index]
        index_name = os.path.basename(file_path)


        
        # Load the .nc file.
        dataset = xr.load_dataset(file_path)
        
        # 
        prompt_kwargs = self.prompt_kwargs.copy() 
        
        
        # Generate the RGB image from selected climate variables.
        rgb_image, [var1, var2, var3] = self.to_image(dataset)  # see function below
        
        # Generate the binary mask from the dataset.
        climatenet_label = prompt_kwargs.pop("climatenet_label", 'cyclone')
        mask = self.get_labels(dataset, label_name=climatenet_label)  # see function below
        
        # Apply optional transforms.
        if self.transforms is not None:
            transformed = self.transforms(image=rgb_image, mask=mask)
            rgb_image, mask = transformed["image"], transformed["mask"]
        
        # Generate prompts (point, box, and noisy masks).
        shot_num = prompt_kwargs.pop("shot_num", None)
        point_coords, box_coords, noisy_object_masks, object_masks = generate_prompts_from_mask(
            gt_mask=mask,
            tgt_prompts=[random.choice(['point', 'box', 'mask'])] if self.train_flag else ['point', 'box'],
            **prompt_kwargs
        )
        
        # Return a dictionary that matches the expected format.
        return {
            "images": rgb_image,  # should be in (H, W, 3) format as a numpy array.
            "gt_masks": mask,     # binary mask.
            "index_name": index_name,
            "point_coords": point_coords,
            "box_coords": box_coords,
            "noisy_object_masks": noisy_object_masks,
            "object_masks": object_masks,
            "var_names": [var1, var2, var3]
        }

    def to_image(self, dataset, var_1='TMQ', var_2='U850', var_3='V850'):
        """
        Convert the dataset into an RGB image using three selected variables.
        """
        # Assume dataset.to_array() gives an array with a "variable" dimension.
        features = dataset.to_array()
        # Select the variables (you may need to adjust this if your dataset is structured differently).
        var1 = features.sel(variable=var_1).values
        var2 = features.sel(variable=var_2).values
        var3 = features.sel(variable=var_3).values

        # Ensure variables are 2D (H, W) before stacking
        var1 = np.squeeze(var1)
        var2 = np.squeeze(var2)
        var3 = np.squeeze(var3)
        
        # Stack the channels to form an RGB image.
        rgb_image = np.stack([var1, var2, var3], axis=-1)
        # Normalize the image to 0-255.
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        rgb_image = (rgb_image * 255).astype(np.uint8)

        # Remove the batch dimension if it exists (1, H, W, C) → (H, W, C)
        if rgb_image.shape[0] == 1:
            rgb_image = np.squeeze(rgb_image, axis=0) 

        return rgb_image, [var1, var2, var3]

    def get_labels(self, dataset, label_name='cyclone'):
        """
        Extract and binarize the segmentation mask from the dataset.
        """
        if label_name == 'cyclone':
            mask_description = 1
        elif label_name == 'river':
            mask_description = 2
        else:
            raise ValueError(f"Unknown label name: {label_name}")
            
        mask = dataset['LABELS'].values
        mask = (mask == mask_description).astype(np.uint8)  # Convert to a binary mask.
        # mask = np.ascontiguousarray(mask)
        # mask = cv2.UMat(mask)  # Ensure the mask is a numpy array
        # print("Mask shape:", mask.shape)
        return mask

    @staticmethod
    def collate_fn(batch):
        # Use the collate function defined in BinaryCATSAMDataset.
        return BinaryCATSAMDataset.collate_fn(batch)
