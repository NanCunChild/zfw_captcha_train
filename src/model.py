# model.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class CRNN(nn.Module):
    def __init__(self, num_chars):
        super(CRNN, self).__init__()
        
        # CNN backbone using ResNet-18
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # Remove AvgPool and FC
        
        # Bidirectional LSTM for sequence modeling
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=False
        )
        
        # Final fully connected layer
        self.fc = nn.Linear(512, num_chars + 1)  # +1 for CTC blank
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize LSTM and FC weights
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # CNN feature extraction
        x = self.cnn(x)  # [batch, 512, height, width]
        
        # Reshape for RNN
        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1)  # [batch, height, width, channels]
        x = x.reshape(batch_size, -1, 512)  # [batch, height*width, channels]
        x = x.permute(1, 0, 2)  # [sequence_length, batch, channels]
        
        # RNN sequence modeling
        x, _ = self.rnn(x)  # [sequence_length, batch, hidden_size*2]
        
        # Classification
        x = self.fc(x)  # [sequence_length, batch, num_classes]
        
        return x