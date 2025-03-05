# dataset.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import random
from torch.utils.data.distributed import DistributedSampler

class CaptchaDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train', train_ratio=0.8):
        self.data_dir = data_dir
        self.transform = transform
        
        # Get all subdirectories (each name is a captcha value)
        all_captcha_dirs = [d for d in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(d)]
        
        # Create a list of tuples (image_path, label)
        self.samples = []
        for captcha_dir in all_captcha_dirs:
            label = os.path.basename(captcha_dir)
            image_files = glob.glob(os.path.join(captcha_dir, '*.png'))
            
            for img_path in image_files:
                self.samples.append((img_path, label))
        
        # Shuffle with a fixed seed for reproducible train/val splits
        random.seed(42)
        random.shuffle(self.samples)
        
        # Split into train and validation sets
        split_idx = int(len(self.samples) * train_ratio)
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:  # 'val'
            self.samples = self.samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load and convert image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders(data_dir, batch_size, num_workers=4, world_size=None, rank=None):
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create datasets
    train_dataset = CaptchaDataset(data_dir, transform=transform, split='train')
    val_dataset = CaptchaDataset(data_dir, transform=transform, split='val')
    
    # Create samplers for distributed training if needed
    train_sampler = None
    val_sampler = None
    
    if world_size is not None and rank is not None:
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=False
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=val_sampler,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler, val_sampler