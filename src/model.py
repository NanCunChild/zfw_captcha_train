# model.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

import config

class CRNN(nn.Module):
    def __init__(self, num_chars):
        super(CRNN, self).__init__()
        self.cnn = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) # Use ResNet-18
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2]) # Remove AvgPool and FC

        self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2)
        self.fc = nn.Linear(512, num_chars + 1)  # +1 for CTC blank

    def forward(self, x):
        # CNN
        x = self.cnn(x)  # [batch, 512, 2, 3]
        x = x.permute(0, 2, 3, 1)  # [batch, 2, 3, 512]
        x = x.reshape(x.size(0), -1, 512) #[batch, 2*3, 512] -> [batch, 6, 512]
        x = x.permute(1, 0, 2)  # [6, batch, 512]
        # RNN
        x, _ = self.rnn(x)  # [6, batch, 512]
        x = self.fc(x)  # [6, batch, num_chars + 1]
        return x