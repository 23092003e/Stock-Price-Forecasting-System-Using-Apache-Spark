import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TimeSeriesLSTM(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def len(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
       
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers ,output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the LSTM layers
        self.lstm = nn.LSTM(input_size = input_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers, 
                            batch_first = True,
                            dropout = 0.2 if num_layers > 1 else 0)
          
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, output_size))
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0,c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        
        return out
