import os
import random
import xarray as xr
import numpy as np
import torch
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

        self.files = [os.path.join(sub_dir, f) for f in sorted(os.listdir(sub_dir)) if f.endswith(".nc")]
        if len(self.files) == 0:
            raise ValueError(f"No .nc files found in directory: {sub_dir}")

        self.train_flag = train_flag
        self.transforms = Compose(transforms) if transforms else None
        
        # Store prompt generation parameters.
        self.prompt_kwargs = prompt_kwargs
        
        shot_num = prompt_kwargs.pop("shot_num", None)
        if shot_num is not None:
            self.files = self.files[:shot_num]
        
        self.mean_std_path = os.path.join(data_dir, "mean_std.npy")
        self.mean_std_dict = self.calculate_mean_std()    

    def calculate_mean_std(self):
        """
        Calculate the mean and std of the data across all the files.
        """
        if os.path.exists(self.mean_std_path):
            print(f"Loading mean/std from {self.mean_std_path}")
            return np.load(self.mean_std_path, allow_pickle=True).item()

        print("Calculating mean/std from scratch...")
        means = []
        stds = []
        
        for file in self.files:
            dataset = xr.load_dataset(file)
            data = dataset.to_array().values.squeeze()
            means.append(np.mean(data, axis=(1,2)))  # Mean for each of the 16 channels
            stds.append(np.std(data, axis=(1,2)))    # Std for each of the 16 channels
        
        # Calculate the overall mean and std for each channel across all files
        mean_dict = np.mean(means, axis=0)
        std_dict = np.mean(stds, axis=0)
        
        result = {"mean": mean_dict, "std": std_dict}
        np.save(self.mean_std_path, result)
        
        # Return a dictionary with channel-wise mean and std
        return result
    
    def z_normalize(self, data):
        """
        Normalize the data using Z-normalization: (X - mean) / std
        """
        mean = self.mean_std_dict["mean"]
        std = self.mean_std_dict["std"]
        
        # Z-normalization for each channel
        normalized_data = (data - mean) / std
        
        return normalized_data
    
    def z_normalize_and_scale(self, data):
        """
        Normalize the data using Z-normalization: (X - mean) / std, then scale it to [0, 255].
        """
        # Z-normalize the data
        mean = self.mean_std_dict["mean"][:, np.newaxis, np.newaxis]
        std = self.mean_std_dict["std"][:, np.newaxis, np.newaxis]
        normalized_data = (data - mean) / std

        # Scale to [0, 255]
        normalized_data_min = normalized_data.min(axis=(0, 1), keepdims=True)
        normalized_data_max = normalized_data.max(axis=(0, 1), keepdims=True)
        
        # Clip values to ensure they stay within the range [0, 1] before multiplying by 255
        scaled_data = np.clip((normalized_data - normalized_data_min) / (normalized_data_max - normalized_data_min), 0, 1) * 255
        
        # Convert to uint8 for image representation
        scaled_data = scaled_data.astype(np.uint8)
        
        return scaled_data

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
        
        # Generate the binary mask from the dataset.
        climatenet_label = prompt_kwargs.pop("climatenet_label", 'cyclone')
        mask = self.get_labels(dataset, label_name=climatenet_label)  # see function below
        data = dataset.to_array().values.squeeze()
        
        # Apply Z-normalization to the data
        data = self.z_normalize_and_scale(data)
        
        rgb_image = self.to_image(dataset, var_1='TMQ', var_2='U850', var_3='V850')
        # Return a dictionary that matches the expected format.
        return {
            "file_name": os.path.splitext(index_name)[0],  # file name without the .nc extension,
            "images": rgb_image,
            "input": data,
            "gt_masks": mask,     # binary mask.
            "index_name": index_name
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

        return rgb_image

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


    @classmethod
    def collate_fn(cls, batch):
        """
        Custom collate function to batch ClimateDataset samples.
        Handles image/mask tensors without assuming same spatial shape.
        """
        batch_dict = {key: [] for key in batch[0].keys()}
        
        while len(batch) != 0:
            ele_dict = batch[0]
            if ele_dict is not None:
                for key in ele_dict.keys():
                    if key not in batch_dict.keys():
                        batch_dict[key] = []
                    batch_dict[key].append(ele_dict[key])
            # remove the redundant data for memory safety
            batch.remove(ele_dict)

        for sample in batch:
            for key, value in sample.items():
                batch_dict[key].append(value)

        # Convert images to tensors: (H, W, 3) → (3, H, W)
        batch_dict['images'] = [
            torch.from_numpy(img).permute(2, 0, 1).float() for img in batch_dict['images']
        ]

        # Convert inputs and masks to tensors
        batch_dict['input'] = torch.stack([torch.from_numpy(inp).float() for inp in batch_dict['input']])

        batch_dict['gt_masks'] = [torch.from_numpy(mask).long() for mask in batch_dict['gt_masks']]

        # Optional: Stack if all are same shape (e.g., during training with fixed size)
        # Otherwise, leave as list to handle variable-sized input
        return batch_dict

