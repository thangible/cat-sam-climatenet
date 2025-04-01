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
        
        shot_num = prompt_kwargs.pop("shot_num", None)
        if shot_num is not None:
            self.files = self.files[:shot_num]
        
        
        

    def __len__(self):
        return len(self.files)

    # def __getitem__(self, index):
    #     # Use filename as the unique index name.
    #     file_path = self.files[index]
    #     index_name = os.path.basename(file_path)


        
    #     # Load the .nc file.
    #     dataset = xr.load_dataset(file_path)
        
    #     # 
    #     prompt_kwargs = self.prompt_kwargs.copy() 
        
        
    #     # Generate the RGB image from selected climate variables.
    #     rgb_image, [var1, var2, var3] = self.to_image(dataset)  # see function below
        
    #     # Generate the binary mask from the dataset.
    #     climatenet_label = prompt_kwargs.pop("climatenet_label", 'cyclone')
    #     mask = self.get_labels(dataset, label_name=climatenet_label)  # see function below
        
    #     # Apply optional transforms.
    #     if self.transforms is not None:
    #         transformed = self.transforms(image=rgb_image, mask=mask)
    #         rgb_image, mask = transformed["image"], transformed["mask"]
        
    #     # Generate prompts (point, box, and noisy masks).
    #     point_coords, box_coords, noisy_object_masks, object_masks = generate_prompts_from_mask(
    #         gt_mask=mask,
    #         tgt_prompts=[random.choice(['point', 'box', 'mask'])] if self.train_flag else ['point', 'box'],
    #         **prompt_kwargs
    #     )
        
    #     # Return a dictionary that matches the expected format.
    #     return {
    #         "file_name": os.path.splitext(index_name)[0],  # file name without the .nc extension
    #         "images": rgb_image,  # should be in (H, W, 3) format as a numpy array.
    #         "gt_masks": mask,     # binary mask.
    #         "index_name": index_name,
    #         "point_coords": point_coords,
    #         "box_coords": box_coords,
    #         "noisy_object_masks": noisy_object_masks,
    #         "object_masks": object_masks,
    #         "var_names": [var1, var2, var3]
    #     }
    
    def __getitem__(self, index):
        # Use filename as the unique index name.
        file_path = self.files[index]
        index_name = os.path.basename(file_path)

        # Load the .nc file.
        dataset = xr.load_dataset(file_path)
        # 
        prompt_kwargs = self.prompt_kwargs.copy() 
        
        # Generate the binary mask from the dataset.
        climatenet_label = prompt_kwargs.pop("climatenet_label", 'cyclone')
        mask = self.get_labels(dataset, label_name=climatenet_label)  # see function below
        data = self.get_data(dataset)
        # Return a dictionary that matches the expected format.
        return {
            "file_name": os.path.splitext(index_name)[0],  # file name without the .nc extension,
            "input": data,
            "gt_masks": mask,     # binary mask.
            "index_name": index_name
        }
    
    def get_data(self, dataset):
        """
        Convert the dataset into a multi-channel image using all 16 variables.
        Returns:
            image: numpy array of shape (H, W, 16)
            var_names: list of variable names
        """
        # Get the dataset as a variable x height x width array
        features = dataset.to_array()  # shape: (variable, H, W)

        # Get variable names
        var_names = features.variable.values.tolist()

        # Convert to numpy and transpose to (H, W, C)
        # shape: (variable, H, W) → (H, W, variable)
        data = features.values  # shape: (16, H, W)
        data = np.transpose(data, (1, 2, 0))  # shape: (H, W, 16)

        # Normalize each channel individually to [0, 255]
        data_min = data.min(axis=(0, 1), keepdims=True)
        data_max = data.max(axis=(0, 1), keepdims=True)
        data = (data - data_min) / (data_max - data_min + 1e-8)  # add epsilon to avoid division by zero
        data = (data * 255).astype(np.uint8)

        return data


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
