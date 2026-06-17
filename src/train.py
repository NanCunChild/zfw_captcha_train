# train.py
"""Train a captcha CNN model and stream metrics to SwanLab.

The captcha is fixed-length (4 digits, 0-9), so the model is a pure CNN with
4 classification heads and the loss is the sum of 4 ``CrossEntropyLoss``
terms. No CTC, no RNN.

Pick a model size with ``--variant``:

    python src/train.py --variant nano      # < 100 KB
    python src/train.py --variant small     # < 500 KB (default)
    python src/train.py --variant full      # ~ 800 KB

You can also produce several artifacts in a single command:

    # Sequential (each variant uses every available GPU via DDP):
    python src/train.py --variants nano,small,full

    # Fully parallel (one variant per GPU, run concurrently):
    python src/train.py --variants all --parallel-variants
"""

from __future__ import annotations

import argparse
import multiprocessing as py_mp
import os
import random
import sys
import time

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

# SwanLab is imported lazily so the script still runs in environments where
# it isn't installed yet (the user just gets a clear message).
try:
    import swanlab  # type: ignore
except ImportError:  # pragma: no cover - exercised only without swanlab
    swanlab = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds. ``deterministic=True`` disables cudnn benchmarking
    (slower but bit-exact reproducible)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


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


def _swanlab_init(args, model, variant):
    """Initialise a SwanLab run; safe to call only on the main process."""
    if not _swanlab_enabled(args):
        return None

    run_config = {
        'variant': variant,
        'epochs': config.EPOCHS,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'lr_scheduler_step': config.LR_SCHEDULER_STEP,
        'lr_scheduler_gamma': config.LR_SCHEDULER_GAMMA,
        'image_width': config.IMG_WIDTH,
        'image_height': config.IMG_HEIGHT,
        'captcha_length': config.CAPTCHA_LENGTH,
        'num_classes': config.NUM_CHARS,
        'trainable_parameters': count_parameters(model),
        'world_size': args.world_size,
        'patience': args.patience,
        'parallel_variants': args.parallel_variants,
    }

    init_kwargs = dict(
        project=args.swanlab_project,
        experiment_name=args.swanlab_experiment or f'{variant}-{int(time.time())}',
        description=f'CNN captcha training ({variant} variant)',
        config=run_config,
        mode=args.swanlab_mode,
    )
    if args.swanlab_workspace:
        init_kwargs['workspace'] = args.swanlab_workspace

    return swanlab.init(**init_kwargs)


def _apply_perf_knobs(args) -> None:
    """Enable optional performance optimisations that are safe across GPUs."""
    if args.tf32:
        # Allow TF32 on Ampere+ GPUs; ignored on older hardware.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except AttributeError:
            pass


def _make_optimizer(model, lr):
    """Adam with fused kernels when supported (CUDA only, PyTorch 2.0+)."""
    try:
        return optim.Adam(model.parameters(), lr=lr, fused=True)
    except (TypeError, RuntimeError):
        return optim.Adam(model.parameters(), lr=lr)


def _multihead_loss(logits: torch.Tensor, targets: torch.Tensor,
                    criterion: nn.Module) -> torch.Tensor:
    """Sum CE loss across the captcha positions.

    ``logits`` has shape (B, P, C) and ``targets`` has shape (B, P).
    """
    b, p, c = logits.shape
    # Flattening once is faster than a Python loop and mathematically the
    # same as summing the per-position losses then dividing by P. To keep
    # the user-facing behaviour ("sum of 4 CrossEntropy"), we scale by P.
    return criterion(logits.reshape(b * p, c), targets.reshape(b * p)) * p


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    """Return ``(num_fully_correct, batch_size)`` for a multi-head batch."""
    preds = logits.argmax(dim=2)                     # (B, P)
    correct = (preds == targets).all(dim=1).sum().item()
    return int(correct), int(targets.size(0))


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_epoch(model, train_loader, optimizer, criterion, scaler, device,
                char_to_idx, idx_to_char, epoch, logger, swanlab_run, log_global_step,
                use_amp):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.EPOCHS} [Train]')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = encode_labels(labels, char_to_idx, config.CAPTCHA_LENGTH).to(
            device, non_blocking=True
        )

        if use_amp:
            with autocast('cuda'):
                logits = model(images)
                loss = _multihead_loss(logits.float(), targets, criterion)
        else:
            logits = model(images)
            loss = _multihead_loss(logits, targets, criterion)

        # Defensive guard: if loss is non-finite (e.g. due to a degenerate
        # batch), skip this step entirely so a single bad batch can't poison
        # the model weights with NaN forever.
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            if batch_idx % 50 == 0:
                logger.warning(f'Skipping batch {batch_idx}: non-finite loss ({loss.item()})')
            continue

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        c, n = _accuracy(logits.detach(), targets)
        correct += c
        total += n

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
            targets = encode_labels(labels, char_to_idx, config.CAPTCHA_LENGTH).to(
                device, non_blocking=True
            )

            logits = model(images)
            loss = _multihead_loss(logits.float(), targets, criterion)
            running_loss += loss.item()

            preds = decode_predictions(logits, idx_to_char)
            all_preds.extend(preds)
            all_labels.extend(labels)
            c, n = _accuracy(logits, targets)
            correct += c
            total += n

            if batch_idx % 10 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct / max(total, 1):.4f}'})

    val_loss = running_loss / max(len(val_loader), 1)
    val_acc = correct / max(total, 1)
    logger.info(f'Valid - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')
    return val_loss, val_acc, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main worker (one variant, one rank)
# ---------------------------------------------------------------------------

def _train_one_variant(rank, world_size, gpu, variant, args):
    """Run the full training loop for ONE variant in the current process.

    ``world_size > 1`` means we are a DDP rank for that variant.
    """
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

    _apply_perf_knobs(args)

    os.makedirs('logs', exist_ok=True)
    log_name = f'train-{variant}' if rank == 0 else f'train-{variant}-rank{rank}'
    log_file = f'logs/{log_name}.log'
    logger = setup_logger(log_name, log_file)

    # Model
    model = build_model(variant).to(device)

    if rank == 0:
        logger.info(
            f'[{variant}] {count_parameters(model):,} trainable parameters '
            f'(world_size={world_size}, gpu={gpu})'
        )

    use_amp = device.type == 'cuda'
    scaler = GradScaler('cuda', enabled=use_amp)

    if world_size > 1:
        if args.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_kwargs = dict(
            device_ids=[gpu] if gpu is not None else None,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        model = DDP(model, **ddp_kwargs)
        if args.static_graph:
            try:
                model._set_static_graph()
            except Exception:  # noqa: BLE001
                pass

    optimizer = _make_optimizer(model, config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.LR_SCHEDULER_STEP,
        gamma=config.LR_SCHEDULER_GAMMA,
    )
    # Per-position cross-entropy. We pass ``reduction='mean'`` and multiply by
    # the number of positions inside ``_multihead_loss`` so the gradient is
    # equivalent to the sum of 4 independent CE losses.
    criterion = nn.CrossEntropyLoss(reduction='mean')

    char_to_idx = {ch: i for i, ch in enumerate(config.CHARS)}
    idx_to_char = {i: ch for i, ch in enumerate(config.CHARS)}

    train_loader, val_loader, train_sampler, val_sampler = get_data_loaders(
        config.DATA_DIR,
        config.BATCH_SIZE,
        num_workers=args.num_workers,
        world_size=world_size if world_size > 1 else None,
        rank=rank if world_size > 1 else None,
        prefetch_factor=args.prefetch_factor,
    )

    # Per-variant checkpoint dir
    ckpt_dir = config.variant_checkpoint_dir(variant)
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)

    # SwanLab on rank-0 only
    swanlab_run = None
    if rank == 0:
        if swanlab is None and args.swanlab_mode != 'disabled':
            logger.warning(
                'swanlab is not installed. Run `pip install swanlab` to enable '
                'experiment tracking, or pass --swanlab-mode disabled to silence.'
            )
        swanlab_run = _swanlab_init(args, model, variant)
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

    # Training loop with optional early stopping.
    global_step = 0
    epochs_without_improvement = 0
    stop_signal = torch.zeros(1, device=device)

    for epoch in range(start_epoch, config.EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        train_loss, train_acc, num_train_batches = train_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, char_to_idx, idx_to_char, epoch, logger,
            swanlab_run, global_step, use_amp,
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
                if epoch % 5 == 0 or epoch == config.EPOCHS - 1:
                    try:
                        sample_img = visualize_model_predictions(
                            model.module if hasattr(model, 'module') else model,
                            val_loader, idx_to_char, device,
                        )
                        swanlab_run.log(
                            {'val/predictions': swanlab.Image(sample_img, caption=f'epoch {epoch + 1}')},
                            step=global_step,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(f'Failed to log prediction image: {exc}')

            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                logger.info(f'New best model with validation accuracy: {val_acc:.4f}')
            else:
                epochs_without_improvement += 1
                logger.info(
                    f'No val_acc improvement for {epochs_without_improvement} epoch(s) '
                    f'(best={best_val_acc:.4f})'
                )

            state = {
                'epoch': epoch + 1,
                'variant': variant,
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

            # Early stopping decision (rank-0 only). Broadcast to all ranks.
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                logger.info(
                    f'Early stopping triggered: {epochs_without_improvement} epochs '
                    f'without improvement (patience={args.patience}).'
                )
                stop_signal.fill_(1)

        global_step += num_train_batches

        # Synchronise the early-stop signal across all DDP ranks.
        if world_size > 1:
            dist.broadcast(stop_signal, src=0)
        if stop_signal.item() > 0:
            break

    # Finalise (rank 0 only)
    if rank == 0:
        logger.info(f'[{variant}] Training completed. Best val acc: {best_val_acc:.4f}')
        final_path = config.final_model_path(variant)
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        torch.save(
            model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            final_path,
        )
        logger.info(f'[{variant}] Final model saved to {final_path} ({_file_size(final_path)})')
        if swanlab_run is not None:
            swanlab_run.log({'final/best_val_accuracy': best_val_acc})
            swanlab_run.finish()

    if world_size > 1:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------

def _ddp_spawn_worker(gpu, ngpus_per_node, args, variant):
    """Top-level worker for mp.spawn (must be picklable)."""
    rank = args.node_rank * ngpus_per_node + gpu
    _train_one_variant(rank, args.world_size, gpu, variant, args)


def _ddp_launch_one_variant(variant: str, args, ngpus_per_node: int) -> None:
    """Train a single variant using DDP across all available GPUs."""
    args.world_size = max(args.nodes * ngpus_per_node, 1)

    if ngpus_per_node <= 1 or args.world_size == 1:
        gpu = 0 if ngpus_per_node >= 1 else None
        _train_one_variant(rank=0, world_size=1, gpu=gpu, variant=variant, args=args)
        return

    mp.spawn(
        _ddp_spawn_worker,
        nprocs=ngpus_per_node,
        args=(ngpus_per_node, args, variant),
    )


def _parallel_pool_worker(gpu_id, variants_to_run, args):
    """One process per GPU; trains every variant in its slice sequentially.

    This is simpler than a queue-based pool but achieves the same effect since
    we pre-shard the variants list across GPUs in round-robin order.
    """
    # Each subprocess gets its own world: single GPU, no DDP.
    args.world_size = 1
    for variant in variants_to_run:
        # Each variant gets its own SwanLab run (init/finish inside).
        _train_one_variant(rank=0, world_size=1, gpu=gpu_id, variant=variant, args=args)


def _parallel_launch_variants(variants, args, ngpus_per_node: int) -> None:
    """Spawn one process per GPU; each process handles its share of variants."""
    if ngpus_per_node <= 1:
        # Falls back to sequential single-GPU.
        for v in variants:
            _ddp_launch_one_variant(v, args, ngpus_per_node)
        return

    # Round-robin: variant i goes to GPU (i % ngpus_per_node).
    shards = [[] for _ in range(ngpus_per_node)]
    for i, v in enumerate(variants):
        shards[i % ngpus_per_node].append(v)

    ctx = py_mp.get_context('spawn')
    procs = []
    for gpu_id, shard in enumerate(shards):
        if not shard:
            continue
        p = ctx.Process(target=_parallel_pool_worker, args=(gpu_id, shard, args))
        p.start()
        procs.append(p)

    exit_code = 0
    for p in procs:
        p.join()
        if p.exitcode != 0:
            exit_code = p.exitcode
    if exit_code != 0:
        sys.exit(exit_code)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_variants(args) -> list[str]:
    if args.variants:
        out = []
        for v in args.variants.split(','):
            v = v.strip().lower()
            if not v:
                continue
            if v == 'all':
                out.extend(VARIANTS)
            elif v in VARIANTS:
                out.append(v)
            else:
                raise SystemExit(f'Unknown variant: {v!r}. Expected one of {VARIANTS} (or "all").')
        # Dedup while preserving order.
        seen = set()
        out = [v for v in out if not (v in seen or seen.add(v))]
        if out:
            return out
    return [args.variant]


def main():
    parser = argparse.ArgumentParser(description='Train a captcha recognition model with SwanLab tracking')

    # Variant selection
    parser.add_argument('--variant', default=config.DEFAULT_VARIANT, choices=VARIANTS,
                        help='Single model size variant. Ignored if --variants is given.')
    parser.add_argument('--variants', default='',
                        help='Comma-separated list of variants to train, or "all". '
                             'Overrides --variant when provided. '
                             'Example: --variants nano,small,full')
    parser.add_argument('--parallel-variants', action='store_true',
                        help='Train multiple variants concurrently, one per GPU. '
                             'Each variant uses a single GPU (no DDP). '
                             'Default is sequential, where each variant in turn '
                             'uses every available GPU via DDP.')

    # Distributed
    parser.add_argument('--nodes', default=1, type=int, help='Number of distributed nodes')
    parser.add_argument('--node-rank', default=0, type=int, help='Rank of this node')
    parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                        help='URL used to initialise the distributed process group')

    # Misc
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--resume', default='', type=str, help='Path to checkpoint to resume from')

    # Early stopping
    parser.add_argument('--patience', type=int, default=8,
                        help='Stop training if val_acc has not improved for this many epochs. '
                             'Set 0 to disable early stopping.')

    # Performance / DDP knobs
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader num_workers per process')
    parser.add_argument('--prefetch-factor', type=int, default=4,
                        help='DataLoader prefetch_factor per worker')
    parser.add_argument('--no-tf32', dest='tf32', action='store_false',
                        help='Disable TF32 matmul on Ampere+ GPUs (TF32 is on by default).')
    parser.set_defaults(tf32=True)
    parser.add_argument('--deterministic', action='store_true',
                        help='Force cudnn.deterministic=True (slower, bit-exact).')
    parser.add_argument('--no-static-graph', dest='static_graph', action='store_false',
                        help='Disable DDP static_graph optimisation.')
    parser.set_defaults(static_graph=True)
    parser.add_argument('--sync-bn', action='store_true',
                        help='Convert BatchNorm to SyncBatchNorm in DDP (slightly slower; '
                             'helps when per-card batch is small).')

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

    set_seed(args.seed, deterministic=args.deterministic)
    os.makedirs('logs', exist_ok=True)

    variants = _parse_variants(args)
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = max(args.nodes * ngpus_per_node, 1)

    print(
        f'Variants to train: {variants}'
        f' | parallel={args.parallel_variants}'
        f' | gpus={ngpus_per_node}'
        f' | patience={args.patience}'
    )

    if len(variants) > 1 and args.parallel_variants and ngpus_per_node > 1:
        # One process per GPU; each handles a shard of variants.
        _parallel_launch_variants(variants, args, ngpus_per_node)
    else:
        # Sequential: each variant in turn uses every GPU via DDP (or single GPU).
        if args.parallel_variants and len(variants) > 1:
            print(
                'Note: --parallel-variants requested but only one GPU is available. '
                'Falling back to sequential training.'
            )
        for variant in variants:
            _ddp_launch_one_variant(variant, args, ngpus_per_node)


if __name__ == '__main__':
    main()
