# train.py
"""Train a captcha CRNN model and stream metrics to SwanLab.

Replaces the previous TensorBoard + local HTML monitor with the SwanLab
experiment-tracking platform. Pick a model size with ``--variant``:

    python src/train.py --variant tiny      # ~1MB  pth
    python src/train.py --variant small     # ~3MB  pth
    python src/train.py --variant medium    # ~10MB pth
    python src/train.py --variant large     # full ResNet-18 backbone (no cap)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# Add project root to path so ``import config`` works when run from any cwd.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from dataset import get_data_loaders
from model import build_model, count_parameters, VARIANTS
from utils import (
    decode_predictions,
    encode_labels,
    load_checkpoint,
    save_checkpoint,
    setup_logger,
    visualize_model_predictions,
)

# SwanLab is imported lazily so the script still runs in environments where it
# isn't installed yet (the user just gets a clear message).
try:
    import swanlab  # type: ignore
except ImportError:  # pragma: no cover - exercised only without swanlab
    swanlab = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _human_readable_size(num_bytes: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB'):
        if num_bytes < 1024 or unit == 'GB':
            return f'{num_bytes:.2f} {unit}'
        num_bytes /= 1024
    return f'{num_bytes:.2f} GB'


def _file_size(path: str) -> str:
    return _human_readable_size(os.path.getsize(path))


def _swanlab_enabled(args) -> bool:
    return swanlab is not None and args.swanlab_mode != 'disabled'


def _swanlab_init(args, model, num_classes):
    """Initialise a SwanLab run; safe to call only on the main process."""
    if not _swanlab_enabled(args):
        return None

    run_config = {
        'variant': args.variant,
        'epochs': config.EPOCHS,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'lr_scheduler_step': config.LR_SCHEDULER_STEP,
        'lr_scheduler_gamma': config.LR_SCHEDULER_GAMMA,
        'image_width': config.IMG_WIDTH,
        'image_height': config.IMG_HEIGHT,
        'num_classes': num_classes,
        'trainable_parameters': count_parameters(model),
        'world_size': args.world_size,
    }

    init_kwargs = dict(
        project=args.swanlab_project,
        experiment_name=args.swanlab_experiment or f'{args.variant}-{int(time.time())}',
        description=f'CRNN captcha training ({args.variant} variant)',
        config=run_config,
        mode=args.swanlab_mode,
    )
    if args.swanlab_workspace:
        init_kwargs['workspace'] = args.swanlab_workspace

    return swanlab.init(**init_kwargs)


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_epoch(model, train_loader, optimizer, criterion, scaler, device,
                char_to_idx, idx_to_char, epoch, logger, swanlab_run, log_global_step):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.EPOCHS} [Train]')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        encoded_labels, label_lengths = encode_labels(labels, char_to_idx)
        encoded_labels = encoded_labels.to(device, non_blocking=True)

        with autocast('cuda'):
            outputs = model(images)

        # CTC loss must run in FP32 to avoid numerical overflow / nan.
        log_probs = torch.nn.functional.log_softmax(outputs.float(), dim=2)
        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=outputs.size(0),
            dtype=torch.long,
            device=device,
        )
        loss = criterion(log_probs, encoded_labels, input_lengths, label_lengths)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # Gradient clipping to prevent exploding gradients with CTC.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        preds = decode_predictions(outputs, idx_to_char)
        for p, t in zip(preds, labels):
            correct += int(p == t)
            total += 1

        if batch_idx % 10 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct / max(total, 1):.4f}'})

        # Per-step swanlab logging (low frequency to avoid noise).
        if swanlab_run is not None and batch_idx % 50 == 0:
            swanlab_run.log(
                {
                    'train/step_loss': loss.item(),
                    'train/running_acc': correct / max(total, 1),
                },
                step=log_global_step + batch_idx,
            )

    train_loss = running_loss / max(len(train_loader), 1)
    train_acc = correct / max(total, 1)
    logger.info(f'Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')
    return train_loss, train_acc, len(train_loader)


def validate_epoch(model, val_loader, criterion, device, char_to_idx, idx_to_char, epoch, logger):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[str] = []
    all_labels: list[str] = []

    pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{config.EPOCHS} [Valid]')
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            encoded_labels, label_lengths = encode_labels(labels, char_to_idx)
            encoded_labels = encoded_labels.to(device, non_blocking=True)

            outputs = model(images)
            log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=outputs.size(0),
                dtype=torch.long,
                device=device,
            )
            loss = criterion(log_probs, encoded_labels, input_lengths, label_lengths)
            running_loss += loss.item()

            preds = decode_predictions(outputs, idx_to_char)
            all_preds.extend(preds)
            all_labels.extend(labels)
            for p, t in zip(preds, labels):
                correct += int(p == t)
                total += 1

            if batch_idx % 10 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct / max(total, 1):.4f}'})

    val_loss = running_loss / max(len(val_loader), 1)
    val_acc = correct / max(total, 1)
    logger.info(f'Valid - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')
    return val_loss, val_acc, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------------

def main_worker(gpu, ngpus_per_node, args):
    rank = args.node_rank * ngpus_per_node + (gpu if gpu is not None else 0)
    world_size = args.world_size

    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method=args.dist_url,
            world_size=world_size,
            rank=rank,
        )

    if gpu is None:
        device = torch.device('cpu')
    else:
        torch.cuda.set_device(gpu)
        device = torch.device(f'cuda:{gpu}')

    # Logger
    os.makedirs('logs', exist_ok=True)
    if rank == 0:
        logger = setup_logger('train', 'logs/train.log')
    else:
        logger = setup_logger(f'train_rank{rank}', f'logs/train_rank{rank}.log')

    # Model
    num_classes = config.NUM_CHARS + 1  # + CTC blank
    model = build_model(args.variant, num_classes=num_classes,
                        pretrained=getattr(args, 'pretrained', True)).to(device)
    if rank == 0:
        logger.info(
            f'Building "{args.variant}" model with '
            f'{count_parameters(model):,} trainable parameters'
        )

    scaler = GradScaler('cuda', enabled=device.type == 'cuda')

    if world_size > 1:
        model = DDP(model, device_ids=[gpu] if gpu is not None else None)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.LR_SCHEDULER_STEP,
        gamma=config.LR_SCHEDULER_GAMMA,
    )
    criterion = nn.CTCLoss(blank=config.NUM_CHARS, reduction='mean')

    char_to_idx = {ch: i for i, ch in enumerate(config.CHARS)}
    idx_to_char = {i: ch for i, ch in enumerate(config.CHARS)}
    idx_to_char[config.NUM_CHARS] = ''  # blank

    train_loader, val_loader, train_sampler, val_sampler = get_data_loaders(
        config.DATA_DIR,
        config.BATCH_SIZE,
        num_workers=4,
        world_size=world_size if world_size > 1 else None,
        rank=rank if world_size > 1 else None,
    )

    # Per-variant checkpoint dir
    ckpt_dir = config.variant_checkpoint_dir(args.variant)
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)

    # SwanLab on main process only
    swanlab_run = None
    if rank == 0:
        if swanlab is None and args.swanlab_mode != 'disabled':
            logger.warning(
                'swanlab is not installed. Run `pip install swanlab` to enable '
                'experiment tracking, or pass --swanlab-mode disabled to silence.'
            )
        swanlab_run = _swanlab_init(args, model, num_classes)
        if swanlab_run is not None:
            logger.info(f'SwanLab tracking enabled (mode={args.swanlab_mode}).')
        else:
            logger.info('SwanLab tracking disabled.')

    # Resume
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f'Loading checkpoint from {args.resume}')
            checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=device)
            if checkpoint:
                start_epoch = checkpoint.get('epoch', 0)
                best_val_acc = checkpoint.get('best_val_acc', 0.0)
                logger.info(
                    f'Resumed at epoch {start_epoch} (best val acc {best_val_acc:.4f})'
                )
        else:
            logger.warning(f'No checkpoint found at {args.resume}')

    # Training loop
    global_step = 0
    for epoch in range(start_epoch, config.EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        train_loss, train_acc, num_train_batches = train_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, char_to_idx, idx_to_char, epoch, logger, swanlab_run, global_step,
        )
        val_loss, val_acc, _, _ = validate_epoch(
            model, val_loader, criterion, device, char_to_idx, idx_to_char, epoch, logger,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        if rank == 0:
            if swanlab_run is not None:
                swanlab_run.log(
                    {
                        'epoch': epoch + 1,
                        'train/loss': train_loss,
                        'train/accuracy': train_acc,
                        'val/loss': val_loss,
                        'val/accuracy': val_acc,
                        'learning_rate': current_lr,
                    },
                    step=global_step,
                )

                # Periodic prediction visualisation.
                if epoch % 5 == 0 or epoch == config.EPOCHS - 1:
                    try:
                        sample_img = visualize_model_predictions(
                            model.module if hasattr(model, 'module') else model,
                            val_loader,
                            idx_to_char,
                            device,
                        )
                        swanlab_run.log(
                            {'val/predictions': swanlab.Image(sample_img, caption=f'epoch {epoch + 1}')},
                            step=global_step,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(f'Failed to log prediction image: {exc}')

            is_best = val_acc > best_val_acc
            best_val_acc = max(val_acc, best_val_acc)
            state = {
                'epoch': epoch + 1,
                'variant': args.variant,
                'model_state_dict': (
                    model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                ),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
            }
            save_checkpoint(state, is_best, ckpt_dir)
            if is_best:
                logger.info(f'New best model with validation accuracy: {val_acc:.4f}')

        global_step += num_train_batches

    # Finalise
    if rank == 0:
        logger.info('Training completed!')

        # Save the final pure-weights pth (for inference / deployment).
        final_path = config.final_model_path(args.variant)
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        torch.save(
            model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            final_path,
        )
        logger.info(f'Final model saved to {final_path} ({_file_size(final_path)})')

        if swanlab_run is not None:
            swanlab_run.log({'final/best_val_accuracy': best_val_acc})
            swanlab_run.finish()

    if world_size > 1:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train a captcha recognition model with SwanLab tracking')
    parser.add_argument('--variant', default=config.DEFAULT_VARIANT, choices=VARIANTS,
                        help='Model size variant (tiny ~1MB, small ~3MB, medium ~10MB, large unlimited)')
    parser.add_argument('--nodes', default=1, type=int, help='Number of distributed nodes')
    parser.add_argument('--node-rank', default=0, type=int, help='Rank of this node')
    parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                        help='URL used to initialise the distributed process group')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--resume', default='', type=str, help='Path to checkpoint to resume from')

    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        help='Skip downloading ImageNet-pretrained ResNet-18 weights (large variant only).')
    parser.set_defaults(pretrained=True)

    # SwanLab options
    parser.add_argument('--swanlab-project', default=config.SWANLAB_PROJECT,
                        help='SwanLab project name')
    parser.add_argument('--swanlab-workspace', default=config.SWANLAB_WORKSPACE,
                        help='SwanLab workspace (organisation) name; optional')
    parser.add_argument('--swanlab-experiment', default=None,
                        help='SwanLab experiment / run name (defaults to "<variant>-<timestamp>")')
    parser.add_argument('--swanlab-mode', default=config.SWANLAB_MODE,
                        choices=('cloud', 'local', 'offline', 'disabled'),
                        help='SwanLab logging mode')

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs('logs', exist_ok=True)

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = max(args.nodes * ngpus_per_node, 1)

    if ngpus_per_node <= 1 or args.world_size == 1:
        gpu = 0 if ngpus_per_node >= 1 else None
        main_worker(gpu, max(ngpus_per_node, 1), args)
    else:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


if __name__ == '__main__':
    main()
