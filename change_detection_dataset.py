"""
Change Detection Dataset Loader
Supports common change detection dataset formats
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms


class ChangeDetectionDataset(Dataset):
    """
    Generic change detection dataset loader
    
    Supports multiple dataset formats:
    1. Separate folders: A_folder, B_folder, label_folder
    2. Paired naming: same filename in different folders
    """
    
    def __init__(self, 
                 root_dir,
                 imgA_dir='A',
                 imgB_dir='B', 
                 label_dir='label',
                 transform=None,
                 label_transform=None,
                 img_extensions=['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']):
        """
        Args:
            root_dir: Root directory of the dataset
            imgA_dir: Subdirectory name for time A images (relative to root_dir)
            imgB_dir: Subdirectory name for time B images (relative to root_dir)
            label_dir: Subdirectory name for labels (relative to root_dir)
            transform: Transform to apply to images
            label_transform: Transform to apply to labels
            img_extensions: List of image file extensions to look for
        """
        self.root_dir = root_dir
        self.imgA_dir = os.path.join(root_dir, imgA_dir) if imgA_dir else None
        self.imgB_dir = os.path.join(root_dir, imgB_dir) if imgB_dir else None
        self.label_dir = os.path.join(root_dir, label_dir) if label_dir else None
        self.transform = transform
        self.label_transform = label_transform
        
        # Get list of image files
        if self.imgA_dir and os.path.exists(self.imgA_dir):
            self.img_files = self._get_image_files(self.imgA_dir, img_extensions)
        elif self.imgB_dir and os.path.exists(self.imgB_dir):
            self.img_files = self._get_image_files(self.imgB_dir, img_extensions)
        else:
            raise ValueError(f"Neither {self.imgA_dir} nor {self.imgB_dir} exists")
        
        print(f"Found {len(self.img_files)} image pairs in {root_dir}")
    
    def _get_image_files(self, directory, extensions):
        """Get list of image files from directory"""
        files = []
        for ext in extensions:
            files.extend([f for f in os.listdir(directory) if f.lower().endswith(ext)])
        return sorted(files)
    
    def _load_image(self, path):
        """Load image from path"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        
        img = Image.open(path).convert('RGB')
        return img
    
    def _load_label(self, path):
        """Load label from path"""
        if not os.path.exists(path):
            # Return zero label if label file doesn't exist
            return Image.new('L', (256, 256), 0)
        
        label = Image.open(path)
        
        # Convert to grayscale if needed
        if label.mode != 'L':
            label = label.convert('L')
        
        # Convert to binary (0 or 1)
        label_array = np.array(label)
        if label_array.max() > 1:
            # Normalize to 0-1 range
            label_array = label_array / 255.0
        
        # Binarize (threshold at 0.5): 255 -> 1, 0 -> 0
        label_array = (label_array > 0.5).astype(np.float32)
        # Keep as 0 or 1 (not 0 or 255)
        label = Image.fromarray((label_array * 255).astype(np.uint8), mode='L')
        
        return label
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        filename = self.img_files[idx]
        
        # Load image A
        if self.imgA_dir:
            imgA_path = os.path.join(self.imgA_dir, filename)
            imgA = self._load_image(imgA_path)
        else:
            raise ValueError("imgA_dir not specified")
        
        # Load image B
        if self.imgB_dir:
            imgB_path = os.path.join(self.imgB_dir, filename)
            imgB = self._load_image(imgB_path)
        else:
            raise ValueError("imgB_dir not specified")
        
        # Load label
        if self.label_dir:
            label_path = os.path.join(self.label_dir, filename)
            label = self._load_label(label_path)
        else:
            # Create dummy label if label_dir not provided
            label = Image.new('L', imgA.size, 0)
        
        # Apply transforms
        if self.transform:
            # Apply same transform to both images
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
        
        if self.label_transform:
            label = self.label_transform(label)
            # Ensure label is binary [0, 1] after transform (convert 255 to 1)
            if isinstance(label, torch.Tensor):
                label = (label > 0.5).float()
                if len(label.shape) == 3:
                    label = label.squeeze(0)
        else:
            # Default: convert to tensor and normalize to [0, 1]
            label = transforms.ToTensor()(label)
            # Convert 255 to 1, 0 to 0 (binary [0, 1])
            label = (label > 0.5).float().squeeze(0)  # Convert to binary [0, 1]
        
        return imgA, imgB, label


def get_default_transform(img_size=256, is_training=False):
    """
    Get default transform for change detection images
    
    Args:
        img_size: Target image size. If None, keep original size (no resize)
        is_training: Whether this is for training (includes augmentation)
    
    Returns:
        Transform function
    """
    transform_list = []
    
    # Only resize if img_size is specified
    if img_size is not None:
        transform_list.append(transforms.Resize((img_size, img_size)))
    
    if is_training:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform = transforms.Compose(transform_list)
    return transform


def get_label_transform(img_size=256):
    """
    Get default transform for labels
    
    Args:
        img_size: Target image size. If None, keep original size (no resize)
    
    Returns:
        Transform function
    """
    transform_list = []
    
    # Only resize if img_size is specified
    if img_size is not None:
        transform_list.append(transforms.Resize((img_size, img_size), interpolation=Image.NEAREST))
    
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def create_change_detection_dataloader(root_dir,
                                      imgA_dir='A',
                                      imgB_dir='B',
                                      label_dir='label',
                                      batch_size=4,
                                      shuffle=False,
                                      num_workers=4,
                                      img_size=256,
                                      is_training=False):
    """
    Create a DataLoader for change detection dataset
    
    Args:
        root_dir: Root directory of the dataset
        imgA_dir: Subdirectory name for time A images
        imgB_dir: Subdirectory name for time B images
        label_dir: Subdirectory name for labels
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes
        img_size: Target image size. If None, keep original size (no resize)
        is_training: Whether this is for training
    
    Returns:
        DataLoader instance
    """
    """
    Create a DataLoader for change detection dataset
    
    Args:
        root_dir: Root directory of the dataset
        imgA_dir: Subdirectory name for time A images
        imgB_dir: Subdirectory name for time B images
        label_dir: Subdirectory name for labels
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes
        img_size: Target image size
        is_training: Whether this is for training
    
    Returns:
        DataLoader instance
    """
    transform = get_default_transform(img_size, is_training)
    label_transform = get_label_transform(img_size)
    
    dataset = ChangeDetectionDataset(
        root_dir=root_dir,
        imgA_dir=imgA_dir,
        imgB_dir=imgB_dir,
        label_dir=label_dir,
        transform=transform,
        label_transform=label_transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader


if __name__ == "__main__":
    # Test dataset loader
    print("Testing ChangeDetectionDataset...")
    
    # Test with dummy data structure (if exists)
    # You can test with your actual dataset path
    test_root = "./test_data"  # Change this to your dataset path
    
    if os.path.exists(test_root):
        dataloader = create_change_detection_dataloader(
            root_dir=test_root,
            batch_size=2,
            img_size=256
        )
        
        print(f"Dataset size: {len(dataloader.dataset)}")
        
        # Test loading one batch
        for imgA, imgB, label in dataloader:
            print(f"Image A shape: {imgA.shape}")
            print(f"Image B shape: {imgB.shape}")
            print(f"Label shape: {label.shape}")
            break
    else:
        print(f"Test directory {test_root} not found. Please provide a valid dataset path.")

