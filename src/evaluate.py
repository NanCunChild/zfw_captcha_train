# evaluate.py
"""Evaluate a trained captcha model on the validation split.

Usage:
    python src/evaluate.py --variant small --model-path checkpoints/small/final_model.pth
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from dataset import get_data_loaders
from model import VARIANTS, build_model
from utils import decode_predictions


def evaluate_model(variant: str, model_path: str, data_dir: str, batch_size: int) -> float:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(variant)

    state_dict = torch.load(model_path, map_location=device)
    # Support both raw state_dicts and full checkpoint dicts.
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.to(device).eval()

    _, val_loader, _, _ = get_data_loaders(data_dir, batch_size, num_workers=4)

    idx_to_char = {i: ch for i, ch in enumerate(config.CHARS)}

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            predictions = decode_predictions(outputs, idx_to_char)
            for p, t in zip(predictions, labels):
                correct += int(p == t)
                total += 1

    accuracy = correct / max(total, 1)
    print(f'Variant: {variant}')
    print(f'Model:   {model_path}')
    print(f'Samples: {total}')
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Evaluate a captcha recognition model.')
    parser.add_argument('--variant', default=config.DEFAULT_VARIANT, choices=VARIANTS,
                        help='Model size variant (must match the trained checkpoint).')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to the model .pth (defaults to the variant best checkpoint).')
    parser.add_argument('--data-dir', type=str, default=config.DATA_DIR,
                        help='Root data directory.')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                        help='Evaluation batch size.')
    args = parser.parse_args()

    model_path = args.model_path or config.best_model_path(args.variant)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f'Model checkpoint not found: {model_path}')

    evaluate_model(args.variant, model_path, args.data_dir, args.batch_size)


if __name__ == '__main__':
    main()
