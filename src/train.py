# train.py
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time
import random
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from dataset import get_data_loaders
from model import CRNN
from utils import (
    encode_labels, decode_predictions, setup_logger,
    save_checkpoint, load_checkpoint, count_parameters,
    visualize_model_predictions
)
from web_monitor import start_monitoring_server

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, train_loader, optimizer, criterion, scaler, device, char_to_idx, epoch, logger):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    train_loader_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
    
    for batch_idx, (images, labels) in enumerate(train_loader_iter):
        # Move data to device
        images = images.to(device, non_blocking=True)
        encoded_labels, label_lengths = encode_labels(labels, char_to_idx)
        encoded_labels = encoded_labels.to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        with autocast():
            outputs = model(images)
            log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
            
            # Calculate input lengths for CTC
            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=outputs.size(0),
                dtype=torch.long,
                device=device
            )
            
            # Calculate loss
            loss = criterion(log_probs, encoded_labels, input_lengths, label_lengths)
        
        # Backward pass with gradient scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        
                # Apply gradients 
        scaler.step(optimizer)
        scaler.update()
        
        # Update statistics
        running_loss += loss.item()
        
        # Calculate accuracy
        preds = decode_predictions(outputs, {v: k for k, v in char_to_idx.items()})
        for i in range(len(preds)):
            if preds[i] == labels[i]:
                correct_predictions += 1
            total_samples += 1
            
        # Update progress bar
        if batch_idx % 10 == 0:
            train_loader_iter.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct_predictions/total_samples:.4f}'
            })
    
    # Calculate epoch statistics
    train_loss = running_loss / len(train_loader)
    train_acc = correct_predictions / total_samples
    
    # Log training statistics
    logger.info(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    
    return train_loss, train_acc

def validate_epoch(model, val_loader, criterion, device, char_to_idx, epoch, logger):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    
    val_loader_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Valid]")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader_iter):
            # Move data to device
            images = images.to(device, non_blocking=True)
            encoded_labels, label_lengths = encode_labels(labels, char_to_idx)
            encoded_labels = encoded_labels.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
            
            # Calculate input lengths for CTC
            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=outputs.size(0),
                dtype=torch.long,
                device=device
            )
            
            # Calculate loss
            loss = criterion(log_probs, encoded_labels, input_lengths, label_lengths)
            
            # Update statistics
            running_loss += loss.item()
            
            # Calculate accuracy
            preds = decode_predictions(outputs, {v: k for k, v in char_to_idx.items()})
            all_predictions.extend(preds)
            all_labels.extend(labels)
            
            for i in range(len(preds)):
                if preds[i] == labels[i]:
                    correct_predictions += 1
                total_samples += 1
            
            # Update progress bar
            if batch_idx % 10 == 0:
                val_loader_iter.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct_predictions/total_samples:.4f}'
                })
    
    # Calculate epoch statistics
    val_loss = running_loss / len(val_loader)
    val_acc = correct_predictions / total_samples
    
    # Log validation statistics
    logger.info(f"Valid - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    
    return val_loss, val_acc, all_predictions, all_labels

def main_worker(gpu, ngpus_per_node, args):
    """Main worker function for distributed training"""
    # Setup process group
    rank = args.node_rank * ngpus_per_node + gpu
    world_size = args.world_size
    
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=world_size,
            rank=rank
        )
    
    # Set device
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    
    # Create Logger
    if rank == 0:
        logger = setup_logger("train", "logs/train.log")
        logger.info(f"Starting training with {ngpus_per_node} GPUs per node, {world_size} total")
        logger.info(f"Model has {count_parameters(model):,} trainable parameters")
    else:
        logger = setup_logger(f"train_rank{rank}", f"logs/train_rank{rank}.log")
    
    # Load model
    model = CRNN(num_chars=config.NUM_CHARS + 1)  # +1 for CTC blank
    model = model.to(device)
    
    # Use mixed precision
    scaler = GradScaler()
    
    # Wrap model with DDP if using distributed training
    if world_size > 1:
        model = DDP(model, device_ids=[gpu])
    
    # Define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.LR_SCHEDULER_STEP,
        gamma=config.LR_SCHEDULER_GAMMA
    )
    criterion = nn.CTCLoss(blank=config.NUM_CHARS, reduction='mean')
    
    # Create character mappings
    char_to_idx = {char: i for i, char in enumerate(config.CHARS)}
    idx_to_char = {i: char for i, char in enumerate(config.CHARS)}
    idx_to_char[config.NUM_CHARS] = ''  # blank character
    
    # Get data loaders for distributed training
    train_loader, val_loader, train_sampler, val_sampler = get_data_loaders(
        config.DATA_DIR,
        config.BATCH_SIZE,
        num_workers=4,
        world_size=world_size if world_size > 1 else None,
        rank=rank if world_size > 1 else None
    )
    
    # Setup tensorboard writer (only on main process)
    if rank == 0:
        tb_writer = SummaryWriter()
        
        # Create directories
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
    
    # Start web monitoring (only on main process)
    if rank == 0 and args.monitor:
        monitoring_thread = start_monitoring_server(
            host=config.MONITOR_HOST, 
            port=config.MONITOR_PORT
        )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint from {args.resume}")
            checkpoint = load_checkpoint(
                args.resume,
                model,
                optimizer,
                scheduler,
                map_location=device
            )
            
            if checkpoint:
                start_epoch = checkpoint.get('epoch', 0)
                best_val_acc = checkpoint.get('best_val_acc', 0.0)
                logger.info(f"Resuming from epoch {start_epoch} with validation accuracy {best_val_acc:.4f}")
        else:
            logger.warning(f"No checkpoint found at {args.resume}")
    
    # Training loop
    for epoch in range(start_epoch, config.EPOCHS):
        # Set the epoch for samplers
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        
        # Train and validate for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scaler, 
            device, char_to_idx, epoch, logger
        )
        
        val_loss, val_acc, val_predictions, val_labels = validate_epoch(
            model, val_loader, criterion, device, char_to_idx, epoch, logger
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to tensorboard (only on main process)
        if rank == 0:
            tb_writer.add_scalar('Loss/train', train_loss, epoch)
            tb_writer.add_scalar('Loss/val', val_loss, epoch)
            tb_writer.add_scalar('Accuracy/train', train_acc, epoch)
            tb_writer.add_scalar('Accuracy/val', val_acc, epoch)
            tb_writer.add_scalar('LearningRate', current_lr, epoch)
            
            # Generate and log sample images with predictions
            if epoch % 5 == 0 or epoch == config.EPOCHS - 1:
                sample_img = visualize_model_predictions(
                    model, val_loader, idx_to_char, device
                )
                tb_writer.add_image('Predictions', np.array(sample_img).transpose(2, 0, 1), epoch)
        
        # Save checkpoint (only on main process)
        if rank == 0:
            is_best = val_acc > best_val_acc
            best_val_acc = max(val_acc, best_val_acc)
            
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc
            }
            
            save_checkpoint(state, is_best, config.CHECKPOINT_DIR)
            
            if is_best:
                logger.info(f"New best model with validation accuracy: {val_acc:.4f}")
    
    # Clean up
    if rank == 0:
        logger.info("Training completed!")
        tb_writer.close()
        
        # Save final model
        final_model_path = os.path.join(config.CHECKPOINT_DIR, 'final_model.pth')
        torch.save(
            model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            final_model_path
        )
        logger.info(f"Final model saved to {final_model_path}")
    
    # Clean up process group
    if world_size > 1:
        dist.destroy_process_group()

def main():
    """Main function to setup and start training"""
    parser = argparse.ArgumentParser(description="Train a captcha recognition model")
    parser.add_argument('--nodes', default=1, type=int, help='Number of nodes')
    parser.add_argument('--node-rank', default=0, type=int, help='Rank of this node')
    parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str, help='URL for distributed training')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--resume', default='', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--monitor', action='store_true', help='Enable web monitoring')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Get the number of GPUs per node
    ngpus_per_node = torch.cuda.device_count()
    
    # Calculate world size
    args.world_size = args.nodes * ngpus_per_node
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Single GPU or CPU training
    if ngpus_per_node == 0 or args.world_size == 1:
        if ngpus_per_node == 0:
            args.device = torch.device('cpu')
            main_worker(None, 1, args)
        else:
            args.device = torch.device('cuda:0')
            main_worker(0, 1, args)
    # Multi-GPU training with DDP
    else:
        # Launch processes
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args)
        )

if __name__ == '__main__':
    main()