# utils.py
import torch
import time
import os
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np

def setup_logger(name, log_file, level=logging.INFO):
    """Setup logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def encode_labels(labels, char_to_idx):
    """Convert text labels to numerical encodings for CTC loss"""
    encoded = []
    lengths = []
    
    for label in labels:
        encoded_label = [char_to_idx[char] for char in label]
        encoded.extend(encoded_label)
        lengths.append(len(encoded_label))
    
    return torch.IntTensor(encoded), torch.IntTensor(lengths)

def decode_predictions(preds, idx_to_char):
    """Decode model predictions to text using CTC decoding"""
    _, max_indices = torch.max(preds, dim=2)  # Get character indices
    decoded_preds = []
    
    for i in range(max_indices.shape[1]):  # iterate over samples in batch
        raw_prediction = max_indices[:, i]
        prediction = []
        
        # CTC decoding: collapse repeated characters and remove blanks
        previous = None
        for j in range(raw_prediction.shape[0]):
            char_index = raw_prediction[j].item()
            if char_index != len(idx_to_char) - 1 and (previous is None or char_index != previous):
                prediction.append(idx_to_char[char_index])
            previous = char_index
        
        decoded_preds.append("".join(prediction))
    
    return decoded_preds

def save_checkpoint(state, is_best, checkpoint_dir):
    """Save model checkpoint"""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    filename = os.path.join(checkpoint_dir, f'checkpoint_epoch_{state["epoch"]}.pth')
    torch.save(state, filename)
    
    if is_best:
        best_filename = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_filename)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, map_location=None):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint

def plot_samples(images, true_labels, pred_labels=None, max_samples=8):
    """Plot sample images with labels for visualization"""
    num_samples = min(len(images), max_samples)
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 3))
    
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        title = f"True: {true_labels[i]}"
        if pred_labels:
            title += f"\nPred: {pred_labels[i]}"
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    return Image.open(buf)

def visualize_model_predictions(model, test_loader, idx_to_char, device, num_samples=8):
    """Generate visualization of model predictions on test data"""
    model.eval()
    images = []
    true_labels = []
    pred_labels = []
    
    with torch.no_grad():
        for img_batch, label_batch in test_loader:
            if len(images) >= num_samples:
                break
                
            img_batch = img_batch.to(device)
            outputs = model(img_batch)
            predictions = decode_predictions(outputs, idx_to_char)
            
            # Add samples to lists
            for i in range(min(len(predictions), num_samples - len(images))):
                images.append(img_batch[i])
                true_labels.append(label_batch[i])
                pred_labels.append(predictions[i])
    
    # Generate plot
    return plot_samples(images, true_labels, pred_labels)

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)