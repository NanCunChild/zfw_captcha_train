import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from dataset import get_data_loader
from model import CRNN
from utils import encode_labels, decode_predictions
    
from tqdm import tqdm


def train_epoch(model, train_loader, optimizer, criterion, device, char_to_idx):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        encoded_labels, label_lengths = encode_labels(labels, char_to_idx)
        encoded_labels = encoded_labels.to(device)

        optimizer.zero_grad()

        outputs = model(images) # Shape: [W', batch_size, num_chars + 1]
        log_probs = torch.log_softmax(outputs, dim=2)

        input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long)

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


# train.py (continued)

def validate_epoch(model, val_loader, criterion, device, char_to_idx):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            encoded_labels, label_lengths = encode_labels(labels, char_to_idx)
            encoded_labels = encoded_labels.to(device)

            outputs = model(images)
            log_probs = torch.log_softmax(outputs, dim=2)

            input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long)
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

def main():
    # 1. Load data
    train_loader, val_loader = get_data_loader(config.DATA_DIR, config.BATCH_SIZE)

    # 2. Create model, optimizer, criterion
    model = CRNN(config.NUM_CHARS).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CTCLoss(blank=config.NUM_CHARS)  # blank is index of blank character

    # Create char_to_idx mapping for encoding.  Important for passing to train/val functions
    char_to_idx = {char: i for i, char in enumerate(config.CHARS)}

    # 3. Setup TensorBoard
    writer = SummaryWriter()

     # Create checkpoint directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)

    # 4. Training loop
    best_val_accuracy = 0.0
    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, config.DEVICE, char_to_idx)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, config.DEVICE, char_to_idx)


        print(f'Epoch {epoch+1}/{config.EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # 5. Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # 6. Save model checkpoint (if improved)
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"Model saved at epoch {epoch+1} with validation accuracy: {best_val_accuracy:.4f}")


    writer.close()

if __name__ == '__main__':
    main()