"""
dataset.py
-----------
Defines utilities for loading and preprocessing 2D medical images
(especially HipMRI NIfTI slices) for training the VQ-VAE model.

Includes:
- to_channels: Converts integer label maps into one-hot channels.
- load_data_2D: Loads and normalizes NIfTI slices into NumPy arrays.
- HipmriDataset: PyTorch `Dataset` wrapper around NIfTI files.
- get_dataloader: Builds a PyTorch `DataLoader` for train/val/test splits.

Author: Youngsu Choi
"""

import numpy as np
import nibabel as nib
import tqdm as tqdm
import torch
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEST = os.path.join(BASE_DIR, "hipmri/keras_slices_test")
DEFAULT_TRAIN = os.path.join(BASE_DIR, "hipmri/keras_slices_train")
DEFAULT_VAL = os.path.join(BASE_DIR, "hipmri/keras_slices_validate")

# Convert integer-labelled image to multi-channel (one-hot encoded) array
def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """
    Convert an integer-labelled image to a one-hot encoded multi-channel array.

    Args:
        arr (np.ndarray): Input array with integer labels (H, W).
        dtype (np.dtype): Output array type.

    Returns:
        np.ndarray: One-hot encoded array with shape (H, W, num_classes).
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res


# Load 2D medical image data
def load_data_2D(imageNames, normImage=False, target_shape=(256, 128), dtype=np.float32, earlyStop=False):
    """
    Load 2D NIfTI medical image slices into NumPy arrays.

    Adapted from COMP3710 A3 Task Sheet.

    Args:
        imageNames (list[str]): List of .nii or .nii.gz image paths.
        normImage (bool): Normalize images to [-1, 1].
        target_shape (tuple): Expected (H, W) for valid slices.
        dtype (np.dtype): Output array dtype.
        earlyStop (bool): If True, stops after ~20 samples (debug mode).

    Returns:
        list[np.ndarray]: List of normalized images shaped (1, H, W).
    """

    images = []

    for i, inName in enumerate(tqdm.tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged') 

        # Deal with images that are not (256, 128) or given target_shape
        if target_shape != inImage.shape:
            continue

        if len(inImage.shape) == 3:
            inImage = inImage[:,:,0] # sometimes extra dims in HipMRI_study data

        inImage = inImage.astype(dtype)

        # Let's normalize to [-1, 1]
        if normImage:
            # * [-1, 1] normalization
            min_val = inImage.min()
            max_val = inImage.max()
            inImage = 2.0 * ((inImage - min_val) / (max_val - min_val + 1e-8)) - 1.0

            # * [0, 1] normalization
            # min_val = inImage.min()
            # max_val = inImage.max()
            # inImage = (inImage - min_val) / (max_val - min_val + 1e-8)

            # * Just normal norm
            # inImage = (inImage - inImage.mean()) / inImage.std()
        
        inImage = inImage[np.newaxis, :, :] # Add Channel dimension
        images.append(inImage)

        if i > 20 and earlyStop:
            break
    
    return images

class HipmriDataset(Dataset):
    """
    Dataset class wrapping HipMRI image slices for PyTorch.

    Each sample is a single-channel tensor (1, H, W).
    """
    def __init__(self,  path=DEFAULT_TRAIN, normImage=True, earlyStop=False, transform=None):
        """
        Args:
            path (str | list[str]): Directory or list of image paths.
            normImage (bool): Normalize to [-1, 1].
            earlyStop (bool): Load fewer samples (debug).
            transform (callable): Optional transform applied per sample.
        """
        self.root = path
        self.normImage = normImage
        self.transform = transform

        # Load Data
        self.images = load_data_2D(
            path,
            normImage=normImage,
            earlyStop=earlyStop
        )

    def __len__(self):
        """Return number of images."""
        return len(self.images)
    
    def __getitem__(self, idx):
        """Return image tensor with optional transform applied."""
        img = self.images[idx]  # shape: (C, H, W)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32)

        return img

def get_dataloader(type="train", batch_size=16, workers=0, earlyStop=False, transform=None):
    """
    Build a PyTorch DataLoader for HipMRI datasets.

    Args:
        type (str): Expects 'train', 'val/validate', or 'test'.
        batch_size (int): Batch size.
        workers (int): DataLoader workers.
        earlyStop (bool): Enable test mode, only loads in 20 images.
        transform (callable): Optional transform applied to samples.

    Returns:
        torch.utils.data.DataLoader: Configured DataLoader.
    """
    
    split_paths = {
        "train": DEFAULT_TRAIN,
        "val": DEFAULT_VAL,
        "validate": DEFAULT_VAL,
        "test": DEFAULT_TEST,
    }

    paths = sorted(glob.glob(os.path.join(split_paths[type], "*.nii.gz")))
    dataset = HipmriDataset(path=paths, normImage=True, earlyStop=earlyStop, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=True)

    return loader

# For SANITY CHECK 
if __name__ == "__main__":

    EARLY_STOP = True
    all_types = ["train", "test", "val"]

    for type in all_types:
        loader = get_dataloader(type, earlyStop=True)

        print("Loader Length: ", len(loader))
        
        for x in loader:
            print("Batch:", x.shape) 
            print(f"Total (Approx) = {len(loader) * x.shape[0]}")
            max_val = x.max().item()
            min_val = x.min().item()
            print(f"Image Max Value: {max_val:.4f}")
            print(f"Image Min Value: {min_val:.4f}")

            # Visual
            imgs = x[:8]  # take first 8 images from the batch
            imgs = imgs.squeeze(1).numpy()  

            fig, axes = plt.subplots(2, 4, figsize=(10, 5))
            for i, ax in enumerate(axes.flat):
                if i < len(imgs):
                    ax.imshow(imgs[i], cmap="gray")
                    ax.set_title(f"Sample {i}")
                    ax.axis("off")

            plt.tight_layout()
            plt.show()
            break
            
