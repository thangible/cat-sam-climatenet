import os
import unittest
from cat_sam.datasets.climatenet import ClimateDataset

class TestClimateDataset(unittest.TestCase):
    def setUp(self):
        # Set up the directory containing the .nc files for testing
        self.data_dir = os.path.join('./data', 'climate')  # Update this path to your actual data directory
        self.dataset = ClimateDataset(data_dir=self.data_dir, train_flag=True)

    def test_dataset_initialization(self):
        # Check if the dataset is initialized correctly
        self.assertGreater(len(self.dataset), 0, "The dataset should contain at least one sample.")
        print(f"Found {len(self.dataset)} files in {self.data_dir}")

    def test_getitem(self):
        # Check if the __getitem__ method works as expected
        sample = self.dataset[0]
        self.assertIn('images', sample, "The sample should contain 'images'.")
        self.assertIn('gt_masks', sample, "The sample should contain 'gt_masks'.")
        self.assertIn('index_name', sample, "The sample should contain 'index_name'.")
        self.assertIn('point_coords', sample, "The sample should contain 'point_coords'.")
        self.assertIn('box_coords', sample, "The sample should contain 'box_coords'.")
        self.assertIn('noisy_object_masks', sample, "The sample should contain 'noisy_object_masks'.")
        self.assertIn('object_masks', sample, "The sample should contain 'object_masks'.")
        print(f"Sample keys: {sample.keys()}")
        

if __name__ == '__main__':
    unittest.main()