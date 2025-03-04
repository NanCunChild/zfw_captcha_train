# dataset.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import config
from utils import encode_labels

class CaptchaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.char_to_idx = {char: i for i, char in enumerate(config.CHARS)}

        for label_dir in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_dir)
            if os.path.isdir(label_path):
                for image_file in os.listdir(label_path):
                    if image_file.endswith('.png'):
                        image_path = os.path.join(label_path, image_file)
                        self.samples.append((image_path, label_dir)) # (image_path, label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB') # Ensure RGB
        if self.transform:
            image = self.transform(image)
        return image, label

def get_data_loader(data_dir, batch_size, train=True):

    transform = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CaptchaDataset(data_dir, transform=transform)

    # Create a sampler for training/validation splitting
    if train:
      train_size = int(0.8 * len(dataset))
      val_size = len(dataset) - train_size
      train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
      val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
      return train_loader, val_loader

    else: # test dataset
      test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
      return test_loader

