# utils.py
import torch

def encode_labels(labels, char_to_idx):
  """将文本标签编码为数字"""
  encoded = []
  lengths = []
  for label in labels:
      encoded_label = [char_to_idx[char] for char in label]
      encoded.extend(encoded_label)
      lengths.append(len(encoded_label))
  return torch.IntTensor(encoded), torch.IntTensor(lengths)

def decode_predictions(preds, idx_to_char):
    """将模型预测解码为文本"""
    _, max_indices = torch.max(preds, dim=2)  # Get character indices
    decoded_preds = []
    for i in range(max_indices.shape[1]): # iterate over samples in batch
        raw_prediction = max_indices[:, i]
        prediction = []

        # CTC decoding: collapse repeated characters and skip blank characters.
        previous = None
        for j in range(raw_prediction.shape[0]):
            char_index = raw_prediction[j].item()
            if char_index != len(idx_to_char) and (previous is None or char_index != previous):
                prediction.append(idx_to_char[char_index])
            previous = char_index

        decoded_preds.append("".join(prediction))
    return decoded_preds