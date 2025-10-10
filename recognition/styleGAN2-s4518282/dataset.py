"""
We assume dataset for ADNI AD / data is within the following path. 
./data/AD/*.png | jpg | jpeg
./data/NC/*.png | jpg | jpeg

Returns (image_tensor, label_int) where labels are:
  NC -> 0, AD -> 1

"""

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt

DEFAULT_PATH = "./data"
CLASS_TO_IDX = {"NC": 0, "AD": 1}
DEFAULT_IMG_SIZE = 256

class ADNIDataset(Dataset):
    """
    Custom data set which returns (image, label) where label is 0 for NC, 1 for AD.
    """

    def __init__(self, root = DEFAULT_PATH, transform = None):
        """
        Args:
            root: directory containing 'AD' and 'NC' subfolders
            transform: torchvision transforms to apply
        """
        self.root = root
        self.transform = transform
        self.items = []

        classes = ["NC", "AD"]
        exts = (".png", ".jpg", ".jpeg")

        # Itereated through folders and append each image long with their labels
        for cls in classes:
            cls_dir = os.path.join(root, cls)
            for f in os.listdir(cls_dir):
                if f.lower().endswith(exts):
                    self.items.append((os.path.join(cls_dir, f), CLASS_TO_IDX[cls]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert("L")  # Grayscale image

        if self.transform is not None:
            img = self.transform(img)
        return img, label

def create_dataloader(image_size=DEFAULT_IMG_SIZE, batch_size=64, num_workers=1):
    """
    Creates dataloader for training. 
    """

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]), # [-1, 1], Nice for GANs...
    ])

    dataset = ADNIDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True)


