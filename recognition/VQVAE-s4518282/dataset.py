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
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res


# Load 2D medical image data
def load_data_2D(imageNames, normImage=False, target_shape=(256, 128), dtype=np.float32, earlyStop=False):
    """
    Load a list of 2D medical image files into a 4D NumPy array.

    REF: This code is taken directly from COMP3710 A3 TASK SHEET

    Parameters:
        imageNames : List of NIfTI file paths (.nii or .nii.gz)
        normImage : If True, normalize each image to zero mean and unit variance.
        categorical : If True, convert label maps into one-hot encoded channels.
        dtype : Desired data type of the output array.
        early_stop : If True, load only a few images (for testing/debugging).

    Returns:
        images: Array of shape (N, H, W, [C])
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

        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        
        inImage = inImage[np.newaxis, :, :] # Add Channel dimension
        images.append(inImage)

        if i > 20 and earlyStop:
            break
    
    return images

class HipmriDataset(Dataset):
    def __init__(self,  path=DEFAULT_TRAIN, normImage=True, earlyStop=False):
        self.root = path
        self.normImage = normImage

        # Load Data
        self.images = load_data_2D(
            path,
            normImage=normImage,
            earlyStop=earlyStop
        )

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]  # shape: (C, H, W)

        # convert to torch tensor
        img = torch.tensor(img, dtype=torch.float32)

        return img

def get_dataloader(type="train", batch_size=16, workers=0, earlyStop=False):
    
    split_paths = {
        "train": DEFAULT_TRAIN,
        "val": DEFAULT_VAL,
        "validate": DEFAULT_VAL,
        "test": DEFAULT_TEST,
    }

    paths = sorted(glob.glob(os.path.join(split_paths[type], "*.nii.gz")))
    dataset = HipmriDataset(path=paths, normImage=True, earlyStop=earlyStop)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=True)

    return loader
    
if __name__ == "__main__":

    EARLY_STOP = True
    all_types = ["train", "test", "val"]

    for type in all_types:
        loader = get_dataloader(type, earlyStop=True)

        print("Loader Length: ", len(loader))
        
        for x in loader:
            print("Batch:", x.shape) 
            print(f"Total (Approx) = {len(loader) * x.shape[0]}")

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
            
