import torch.nn as nn

class SequentialLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = [128, 64, 32], dropout = 0.2):
        super(SequentialLSTM, self).__init__()
        # Define LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_size[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size[0], hidden_size[1], batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size[1], hidden_size[2], batch_first=True)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Define fully connected layers
        self.fc1 = nn.Linear(hidden_size[2], 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_size)
    
    def forward(self, x):
        # LSTM1
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        
        # LSTM2
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        
        # LSTM3
        x, _ = self.lstm3(x)
        
        # Fully connected layers
        x = self.relu(self.fc1(x[:, -1, :]))  
        x = self.dropout(x)
        x = self.fc2(x)
        return x
