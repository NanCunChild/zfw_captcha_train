# evaluate.py

import torch
import config
from dataset import get_data_loader
from model import CRNN
from utils import decode_predictions
from tqdm import tqdm

def evaluate_model(model_path, data_dir, batch_size):
    # Load model
    model = CRNN(config.NUM_CHARS)
    model.load_state_dict(torch.load(model_path))
    model.to(config.DEVICE)
    model.eval()  # Set the model to evaluation mode

    # Get data loader
    test_loader = get_data_loader(data_dir, batch_size, train=False)

    # Create idx_to_char mapping
    idx_to_char = {i: char for i, char in enumerate(config.CHARS)}
    idx_to_char[len(config.CHARS)] = '' # blank character

    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculations during evaluation
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(config.DEVICE)
            # labels are strings; no need to encode for evaluation, just comparison

            # Get model predictions
            outputs = model(images)
            predictions = decode_predictions(outputs, idx_to_char)

            # Compare predictions with ground truth labels
            for i in range(len(predictions)):
                if predictions[i] == labels[i]:
                    correct_predictions += 1
                total_samples += 1

    accuracy = correct_predictions / total_samples
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy


if __name__ == '__main__':
    # Example usage
    model_path = config.MODEL_PATH  # Path to the trained model
    data_dir = config.DATA_DIR     # Path to the test dataset directory
    batch_size = config.BATCH_SIZE
    evaluate_model(model_path, data_dir, batch_size)
