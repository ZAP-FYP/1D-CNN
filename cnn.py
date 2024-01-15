import torch
import torch.nn as nn
import torch.nn.functional as F

class MovingAverage(nn.Module):
    def __init__(self, window_size):
        super(MovingAverage, self).__init__()
        self.window_size = window_size

    def forward(self, x):
        x = F.avg_pool1d(x, kernel_size=self.window_size, stride=1, padding=(self.window_size - 1) // 2)
        return x

class ConvNet(nn.Module):
    
    def __init__(self, in_channels, in_seq_len, ma_window_size):
        super(ConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=15, stride=1, padding=1),
            nn.Conv1d(in_channels=128, out_channels=5, kernel_size=15, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=3, stride=2) 
            # nn.ReLU(),
            # nn.Conv1d(in_channels=20, out_channels=128, kernel_size=3, stride=1, padding=2),
            # nn.ReLU()
        )
        
        self.moving_avg = MovingAverage(ma_window_size)

        # self.globalAvg = nn.AdaptiveAvgPool1d(100)
        # self.flatten = nn.Flatten()
        
        # The formula is (input_size - kernel_size + 2*padding) / stride + 1
        # conv_output_size = (in_seq_len - 3 + 2*2) + 1
        conv_output_size = in_seq_len
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv1d):
                conv_output_size = ((conv_output_size + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0]) + 1
            elif isinstance(layer, nn.MaxPool1d):
                conv_output_size = ((conv_output_size - layer.kernel_size) // layer.stride) + 1

        # conv_out_size = in_seq_len   # Two max pooling layers with kernel size 2
        self.fc_layers = nn.Sequential(
            nn.Linear(5 * conv_output_size, 256),
            nn.Linear(256, 500)

        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.moving_avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
