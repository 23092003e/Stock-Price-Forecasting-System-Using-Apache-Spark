import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1):
        super(LSTMModel, self).__init__()
        
        # LSTM layers with exact sizes from checkpoint
        self.lstm1 = nn.LSTM(input_size, 128, batch_first=True)  # input -> 128
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)         # 128 -> 64
        self.lstm3 = nn.LSTM(64, 32, batch_first=True)          # 64 -> 32
        
        # Fully connected layers with exact sizes from checkpoint
        self.fc1 = nn.Linear(32, 16)  # 32 -> 16
        self.fc2 = nn.Linear(16, 1)   # 16 -> 1
        
        # Dropout layer
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # First LSTM layer
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        
        # Second LSTM layer
        out, _ = self.lstm2(out)
        out = self.dropout(out)
        
        # Third LSTM layer
        out, _ = self.lstm3(out)
        
        # Get the last time step output
        out = out[:, -1, :]
        
        # Fully connected layers
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out 