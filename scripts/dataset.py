from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from PIL import Image

import albumentations as A
from helper import convert_to_binary_mask_2

class SolarTrackerDataset(Dataset):
    def __init__(self, img_paths, mask_paths, augmentation=False):
        self.image_paths = img_paths
        self.mask_paths = mask_paths
        self.augmentation = augmentation
        self.transform = self.get_augmentations() if self.augmentation else None
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img_path, mask_path = "../" + img_path, "../" + mask_path

        img = Image.open(img_path)
        img_np = np.array(img).astype(np.float32)/255.0
    
        height, width, _ = img_np.shape
        pad_height = max(0, 500 - height)
        pad_width = max(0, 500 - width)
        img_np = np.pad(img_np, ((0, pad_height), (0, pad_width), (0, 0)),  mode='constant', constant_values=0)
        mask = convert_to_binary_mask_2(mask_path).astype(np.float32)

        if self.augmentation and self.transform:
            augmented = self.transform(image=img_np, mask=mask)
            img_np, mask = augmented['image'], augmented['mask']

        return idx, img_np, mask

    def get_augmentations(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=10, p=0.7),          # small rotations ±10°
            A.ShiftScaleRotate(
                shift_limit=0.02, scale_limit=0.05, rotate_limit=0, p=0.5
            ),                           
            A.ElasticTransform(alpha=1, sigma=2.0, alpha_affine=0.5, p=0.3),

            # Photometric augmentations
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),

            # Optional occlusion / shadow simulation
            A.CoarseDropout(max_holes=2, max_height=30, max_width=30, min_holes=1, min_height=10, min_width=10, fill_value=0, mask_fill_value=0, p=0.3),

            # using ImageNet normalization since we are using pretrained backbones
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])