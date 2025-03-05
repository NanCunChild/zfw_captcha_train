# train.py
import sys
import os
import argparse # 导入 argparse 模块

# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import config
from dataset import get_data_loader
from model import CRNN
from utils import encode_labels, decode_predictions

def train_epoch(model, train_loader, optimizer, criterion, device, char_to_idx, rank):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # tqdm only on the master process
    train_loader_iter = tqdm(train_loader, desc=f"Training (Rank {rank})", disable=(rank != 0))

    for images, labels in train_loader_iter:
        images = images.to(device)
        encoded_labels, label_lengths = encode_labels(labels, char_to_idx)
        encoded_labels = encoded_labels.to(device)

        optimizer.zero_grad()

        outputs = model(images) # Shape: [W', batch_size, num_chars + 1]
        log_probs = torch.log_softmax(outputs, dim=2)

        input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long, device=device) # input_lengths也要放到GPU上

        loss = criterion(log_probs, encoded_labels, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        predictions = decode_predictions(outputs, {v: k for k, v in char_to_idx.items()})
        for i in range(len(predictions)):
          if predictions[i] == labels[i]:
            correct_predictions += 1
          total_samples += 1

    train_loss = running_loss / len(train_loader)
    train_acc = correct_predictions / total_samples

    return train_loss, train_acc

def validate_epoch(model, val_loader, criterion, device, char_to_idx, rank):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # tqdm only on master process
    val_loader_iter = tqdm(val_loader, desc=f"Validating (Rank {rank})", disable=(rank != 0))

    with torch.no_grad():
        for images, labels in val_loader_iter:
            images = images.to(device)
            encoded_labels, label_lengths = encode_labels(labels, char_to_idx)
            encoded_labels = encoded_labels.to(device)

            outputs = model(images)
            log_probs = torch.log_softmax(outputs, dim=2)

            input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long, device = device) # input_lengths也要放到GPU
            loss = criterion(log_probs, encoded_labels, input_lengths, label_lengths)

            running_loss += loss.item()

             # Calculate accuracy
            predictions = decode_predictions(outputs, {v: k for k, v in char_to_idx.items()})
            for i in range(len(predictions)):
                if predictions[i] == labels[i]:
                    correct_predictions += 1
                total_samples += 1

    val_loss = running_loss / len(val_loader)
    val_accuracy = correct_predictions / total_samples
    return val_loss, val_accuracy

def main(rank, world_size):
    # 1. 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # Use NCCL backend for GPU
    torch.cuda.set_device(rank) # Set device for each process
    device = torch.device(f"cuda:{rank}")

    # 2. Load data (with DistributedSampler)
    train_loader, val_loader = get_data_loader(config.DATA_DIR, config.BATCH_SIZE // world_size) # Divide batch size by world size
    train_sampler = DistributedSampler(train_loader.dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_loader.dataset, num_replicas=world_size, rank=rank)

    # Re-create data loaders with the samplers
    train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=config.BATCH_SIZE // world_size, sampler=train_sampler, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=config.BATCH_SIZE // world_size, sampler=val_sampler, num_workers=4)

    # 3. Create model and wrap with DDP
    model = CRNN(config.NUM_CHARS).to(device)
    model = DDP(model, device_ids=[rank])

    # 4. Optimizer and Criterion
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CTCLoss(blank=config.NUM_CHARS)  # blank is index of blank character

    # Create char_to_idx mapping
    char_to_idx = {char: i for i, char in enumerate(config.CHARS)}

    # 5. Setup TensorBoard (only on rank 0)
    if rank == 0:
        writer = SummaryWriter()

     # Create checkpoint directory
    if rank == 0: # Only create on main process
        os.makedirs("checkpoints", exist_ok=True)

    # 6. Training loop
    best_val_accuracy = 0.0
    for epoch in range(config.EPOCHS):
        train_sampler.set_epoch(epoch)  # Important for shuffling to work correctly
        val_sampler.set_epoch(epoch)

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, char_to_idx, rank)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, char_to_idx, rank)

        # Only print and save on rank 0
        if rank == 0:
            print(f'Epoch {epoch+1}/{config.EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            # Log metrics to TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)

            # Save model checkpoint (if improved)
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(model.module.state_dict(), config.MODEL_PATH) # Save the underlying module
                print(f"Model saved at epoch {epoch+1} with validation accuracy: {best_val_accuracy:.4f}")

    if rank == 0:
        writer.close()

    dist.destroy_process_group() # Clean up

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a captcha recognition model with DDP.")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training.") # 添加参数
    args = parser.parse_args()

    world_size = 4 # Set world size (number of GPUs)
    # Use torchrun to launch
    # The --local_rank argument will be passed automatically by torchrun
    main(args.local_rank, world_size)
