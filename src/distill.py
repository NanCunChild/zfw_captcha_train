# distill.py
"""Knowledge distillation: transfer knowledge from a trained teacher to a
smaller student model.

Usage:
    # Distill small → nano (default)
    python src/distill.py --teacher-path checkpoints/small/final_model.pth

    # Distill small → micro (even smaller than nano)
    python src/distill.py --teacher-path checkpoints/small/final_model.pth --student-variant micro

    # Experiment with custom student channels
    python src/distill.py --teacher-path checkpoints/small/final_model.pth --student-channels 8,16,24
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from dataset import get_data_loaders
from model import (
    CaptchaCNN, build_model, count_parameters, MODEL_CONFIGS, VARIANTS,
    _conv_block, NUM_POSITIONS, NUM_DIGITS,
)
from utils import encode_labels, decode_predictions, setup_logger

try:
    import swanlab
except ImportError:
    swanlab = None


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """Combined KD + CE loss for multi-head captcha model.

    Args:
        student_logits: (B, P, C) student output logits
        teacher_logits: (B, P, C) teacher output logits (detached)
        targets: (B, P) hard labels
        temperature: softmax temperature for KD
        alpha: weight for KD loss (1 - alpha for hard CE)

    Returns:
        Scalar loss tensor.
    """
    b, p, c = student_logits.shape

    student_flat = student_logits.reshape(b * p, c)
    teacher_flat = teacher_logits.reshape(b * p, c)
    targets_flat = targets.reshape(b * p)

    student_log_soft = F.log_softmax(student_flat / temperature, dim=1)
    teacher_soft = F.softmax(teacher_flat / temperature, dim=1)
    kd_loss = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)

    ce_loss = F.cross_entropy(student_flat, targets_flat)

    return alpha * kd_loss + (1.0 - alpha) * ce_loss


def build_custom_student(channels_str: str) -> CaptchaCNN:
    """Build a student with custom channel configuration.

    ``channels_str`` is a comma-separated list of integers, e.g. "8,16,24".
    The first 3 layers get MaxPool, the rest do not (same convention as
    MODEL_CONFIGS).
    """
    channels = [int(x.strip()) for x in channels_str.split(',')]
    pools = [i < 3 for i in range(len(channels))]

    model = CaptchaCNN.__new__(CaptchaCNN)
    nn.Module.__init__(model)

    layers: list[nn.Module] = []
    in_ch = 3
    for out_ch, do_pool in zip(channels, pools):
        layers.append(_conv_block(in_ch, out_ch))
        if do_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        in_ch = out_ch
    model.cnn = nn.Sequential(*layers)

    feat_dim = channels[-1]
    model.collapse = nn.AdaptiveAvgPool2d((1, NUM_POSITIONS))
    model.heads = nn.ModuleList(
        [nn.Linear(feat_dim, NUM_DIGITS) for _ in range(NUM_POSITIONS)]
    )
    model.num_positions = NUM_POSITIONS
    model.num_digits = NUM_DIGITS
    model._initialize_weights()
    return model


def train_distill(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = setup_logger('distill', 'logs/distill.log')

    teacher = build_model(args.teacher_variant).to(device)
    state_dict = torch.load(args.teacher_path, map_location=device)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    teacher.load_state_dict(state_dict)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    logger.info(f'Teacher loaded: {args.teacher_variant} ({count_parameters(teacher):,} params)')

    if args.student_channels:
        student = build_custom_student(args.student_channels).to(device)
        student_name = f'custom-{args.student_channels}'
    else:
        student = build_model(args.student_variant).to(device)
        student_name = args.student_variant
    logger.info(f'Student: {student_name} ({count_parameters(student):,} params)')

    param_bytes = count_parameters(student) * 4
    logger.info(f'Student estimated size: {param_bytes / 1024:.1f} KB')

    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    char_to_idx = {ch: i for i, ch in enumerate(config.CHARS)}
    idx_to_char = {i: ch for i, ch in enumerate(config.CHARS)}

    train_loader, val_loader, _, _ = get_data_loaders(
        config.DATA_DIR, args.batch_size, num_workers=4
    )

    swanlab_run = None
    if swanlab is not None and args.swanlab_mode != 'disabled':
        swanlab_run = swanlab.init(
            project=config.SWANLAB_PROJECT,
            experiment_name=f'distill-{student_name}-{int(time.time())}',
            description=f'KD: {args.teacher_variant} → {student_name}',
            config={
                'teacher': args.teacher_variant,
                'student': student_name,
                'student_params': count_parameters(student),
                'temperature': args.temperature,
                'alpha': args.alpha,
                'epochs': args.epochs,
                'lr': args.lr,
                'batch_size': args.batch_size,
            },
            mode=args.swanlab_mode,
        )

    save_dir = os.path.join(config.CHECKPOINT_DIR, f'distill-{student_name}')
    os.makedirs(save_dir, exist_ok=True)

    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        student.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Distill]')
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            targets = encode_labels(labels, char_to_idx, config.CAPTCHA_LENGTH).to(
                device, non_blocking=True
            )

            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)
            loss = distillation_loss(
                student_logits, teacher_logits, targets,
                temperature=args.temperature, alpha=args.alpha,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()
            preds = student_logits.argmax(dim=2)
            c = (preds == targets).all(dim=1).sum().item()
            correct += c
            total += targets.size(0)

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/max(total,1):.4f}'})

        scheduler.step()
        train_loss = running_loss / max(len(train_loader), 1)
        train_acc = correct / max(total, 1)

        student.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                targets = encode_labels(labels, char_to_idx, config.CAPTCHA_LENGTH).to(
                    device, non_blocking=True
                )
                teacher_logits = teacher(images)
                student_logits = student(images)
                loss = distillation_loss(
                    student_logits, teacher_logits, targets,
                    temperature=args.temperature, alpha=args.alpha,
                )
                val_loss += loss.item()
                preds = student_logits.argmax(dim=2)
                c = (preds == targets).all(dim=1).sum().item()
                val_correct += c
                val_total += targets.size(0)

        val_loss /= max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1)

        logger.info(
            f'Epoch {epoch+1}/{args.epochs} - '
            f'train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, '
            f'val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}'
        )

        if swanlab_run is not None:
            swanlab_run.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/accuracy': train_acc,
                'val/loss': val_loss,
                'val/accuracy': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr'],
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(student.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            logger.info(f'New best: {val_acc:.4f}')
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            logger.info(f'Early stopping at epoch {epoch+1} (patience={args.patience})')
            break

    final_path = os.path.join(save_dir, 'final_model.pth')
    torch.save(student.state_dict(), final_path)
    file_size = os.path.getsize(final_path)
    logger.info(
        f'Done. Best val_acc: {best_val_acc:.4f} | '
        f'Final model: {final_path} ({file_size/1024:.1f} KB)'
    )

    if swanlab_run is not None:
        swanlab_run.log({'final/best_val_accuracy': best_val_acc})
        swanlab_run.finish()

    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description='Knowledge distillation for captcha models')

    parser.add_argument('--teacher-variant', default='small', choices=VARIANTS)
    parser.add_argument('--teacher-path', required=True,
                        help='Path to trained teacher model checkpoint')
    parser.add_argument('--student-variant', default='nano', choices=VARIANTS)
    parser.add_argument('--student-channels', default=None,
                        help='Custom student channel config (e.g. "8,16,24" or "6,12,20,28"). '
                             'Overrides --student-variant.')

    parser.add_argument('--temperature', type=float, default=4.0,
                        help='Distillation temperature (higher = softer distribution)')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='Weight for KD loss (1-alpha for hard CE loss)')

    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (0 to disable)')

    parser.add_argument('--swanlab-mode', default=config.SWANLAB_MODE,
                        choices=('cloud', 'local', 'offline', 'disabled'))

    args = parser.parse_args()
    os.makedirs('logs', exist_ok=True)
    train_distill(args)


if __name__ == '__main__':
    main()
