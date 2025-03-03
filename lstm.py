import torch.nn as nn

class SequentialLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super(SequentialLSTM, self).__init__()
        # Define LSTM layers
        self.lstm1 = nn.LSTM(input_size, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.lstm3 = nn.LSTM(64, 32, batch_first=True)
        
        # Define fully connected layers
        self.fc1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_size)
    
    def forward(self, x):
        # LSTM1
        x, _ = self.lstm1(x)
        
        # LSTM2
        x, _ = self.lstm2(x)
        
        # LSTM3
        x, _ = self.lstm3(x)
        
        # Fully connected layers
        x = self.relu(self.fc1(x[:, -1, :]))  
        x = self.fc2(x)
        return x
