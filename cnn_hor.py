
    
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, in_channels, in_seq_len):
        super(ConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels=128, out_channels=5, kernel_size=3, stride=1, padding=1),
        )

        
        # Adjusted the number of units in the fully connected layer
        conv_out_size = in_seq_len   # Two max pooling layers with kernel size 2
        self.fc_layers = nn.Sequential(
            nn.Linear(5 * conv_out_size, 256),
            nn.Linear(256, 500)

        )

        # self.conv_layers = nn.Sequential(
        #     nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=7, stride=1, padding=3),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2),
        #     nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2),
        #     nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2),
        # )
        
        # # Adjusted the number of units in the fully connected layer
        # conv_out_size = in_seq_len // 8  # Three max pooling layers with kernel size 2
        # self.fc_layers = nn.Sequential(
        #     nn.Linear(32 * conv_out_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 500),
        # )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.fc_layers(x)
        return x
