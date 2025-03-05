# evaluate.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import config
from dataset import get_data_loader
from model import CRNN
from utils import decode_predictions
from tqdm import tqdm
import argparse # 导入 argparse 模块
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def evaluate_model(model_path, data_dir, batch_size, rank, world_size):
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Load model
    model = CRNN(config.NUM_CHARS)
    model.load_state_dict(torch.load(model_path, map_location=device)) # 指定map_location
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Get data loader
    test_loader = get_data_loader(data_dir, batch_size, train=False)
    test_sampler = DistributedSampler(test_loader.dataset, num_replicas=world_size, rank=rank, shuffle=False) #注意这里shuffle=False
    test_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4)
    # Create idx_to_char mapping
    idx_to_char = {i: char for i, char in enumerate(config.CHARS)}
    idx_to_char[len(config.CHARS)] = '' # blank character

    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculations during evaluation
        for images, labels in tqdm(test_loader, desc="Evaluating", disable=(rank!=0)):
            images = images.to(device)
            # labels are strings; no need to encode for evaluation, just comparison

            # Get model predictions
            outputs = model(images)
            predictions = decode_predictions(outputs, idx_to_char)

            # Compare predictions with ground truth labels
            for i in range(len(predictions)):
                if predictions[i] == labels[i]:
                    correct_predictions += 1
                total_samples += 1
    
    # Gather results from all processes
    correct_predictions_tensor = torch.tensor(correct_predictions).to(device)
    total_samples_tensor = torch.tensor(total_samples).to(device)
    dist.reduce(correct_predictions_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total_samples_tensor, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
      accuracy = correct_predictions_tensor.item() / total_samples_tensor.item()
      print(f'Test Accuracy: {accuracy:.4f}')
    
    dist.destroy_process_group()
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a captcha recognition model with DDP.")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training.") # 添加参数
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint.")
    args = parser.parse_args()

    world_size = 4
    # Example usage
    model_path = config.MODEL_PATH  # Path to the trained model
    data_dir = config.DATA_DIR     # Path to the test dataset directory
    batch_size = config.BATCH_SIZE // world_size
    evaluate_model(args.model_path, data_dir, batch_size, args.local_rank, world_size)