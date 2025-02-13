import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


class ImageDataset(Dataset):
    """
    Custom Dataset for loading images from a directory.
    """

    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            transform (torchvision.transforms.Compose): Transformations to apply.
        """
        self.image_dir = image_dir
        self.image_paths = sorted([
            os.path.join(image_dir, img)
            for img in os.listdir(image_dir)
            if img.endswith((".png", ".jpg", ".jpeg", ".tif"))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Loads an image and applies the transformation.

        Returns:
            tuple: (image_tensor, img_path)
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_path  # Return both image tensor and path


# Define transformation pipeline
transform_pipeline = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])


def load_images(folder_path, batch_size):
    """
    Loads images from a folder and returns a DataLoader.

    Args:
        folder_path (str): Path to the folder containing images.
        batch_size (int): Batch size for training.

    Returns:
        DataLoader: PyTorch DataLoader with images.
    """
    dataset = ImageDataset(folder_path, transform=transform_pipeline)

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,  # Adjust based on available CPU cores
        pin_memory=True  # Speeds up data transfer to GPU if using CUDA
    )
