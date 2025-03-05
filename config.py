# config.py
import os

# Data paths
DATA_DIR = 'data/captcha_get'

# Image dimensions
IMG_WIDTH = 90
IMG_HEIGHT = 34

# Character set
CHARS = '0123456789'
NUM_CHARS = len(CHARS)

# Training parameters
BATCH_SIZE = 128  # Increased batch size for faster training
EPOCHS = 50
LEARNING_RATE = 0.001
LR_SCHEDULER_STEP = 10
LR_SCHEDULER_GAMMA = 0.5

# Web monitoring
MONITOR_PORT = 8080
MONITOR_HOST = "0.0.0.0"

# Checkpoints
CHECKPOINT_DIR = 'checkpoints'
MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'model.pth')
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')