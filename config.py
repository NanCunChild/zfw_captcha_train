# config.py
import os

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_DIR = 'data/captcha_get'

# Image dimensions (input to the model)
IMG_WIDTH = 90
IMG_HEIGHT = 34

# Character set. The captcha is fixed-length, exactly 4 digits.
CHARS = '0123456789'
NUM_CHARS = len(CHARS)
CAPTCHA_LENGTH = 4

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 0.001
LR_SCHEDULER_STEP = 10
LR_SCHEDULER_GAMMA = 0.5

# ---------------------------------------------------------------------------
# Model variants
# ---------------------------------------------------------------------------
# Available variants (defined in src/model.py). All are pure CNN with 4
# independent classification heads (one per digit position).
#   "micro"  ~  24 KB   distillation target (experimental)
#   "nano"   ~  85 KB   absolute minimum (< 100 KB)
#   "small"  ~ 385 KB   default lightweight
#   "full"   ~ 785 KB
DEFAULT_VARIANT = 'small'
VARIANTS = ('micro', 'nano', 'small', 'full')

# ---------------------------------------------------------------------------
# SwanLab monitoring
# ---------------------------------------------------------------------------
# `mode` may be: "cloud" (default, push to swanlab.cn), "local" (self-hosted
# server -- requires SWANLAB_HOST), "offline" (write logs locally without
# uploading) or "disabled" (no swanlab tracking at all).
SWANLAB_PROJECT = 'zfw_captcha_train'
SWANLAB_WORKSPACE = None        # set to a swanlab workspace name if needed
SWANLAB_MODE = 'cloud'

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = 'checkpoints'


def variant_checkpoint_dir(variant: str) -> str:
    """Per-variant checkpoint directory so different sizes don't collide."""
    return os.path.join(CHECKPOINT_DIR, variant)


def best_model_path(variant: str) -> str:
    return os.path.join(variant_checkpoint_dir(variant), 'best_model.pth')


def final_model_path(variant: str) -> str:
    return os.path.join(variant_checkpoint_dir(variant), 'final_model.pth')


# Backwards-compatible aliases (old code referenced these directly).
MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'model.pth')
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
