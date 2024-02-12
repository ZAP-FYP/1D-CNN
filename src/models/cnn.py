import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, in_channels, in_seq_len):
        super(ConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
        )

        self.globalAvg = nn.AdaptiveAvgPool1d(100)
        # self.flatten = nn.Flatten()

        # self.fc_layers = nn.Sequential(
        #     nn.Linear(100, 100),
        #     nn.ReLU(),
        #     # nn.Linear(50, 500),
        # )

    def forward(self, x):
        x = self.conv_layers(x)
        # x = self.flatten(x)
        # x = torch.flatten(x, 2)
        # x = self.fc_layers(x)
        x = self.globalAvg(x)
        # x = x.view(x.size(0), -1)
        # print(f"Shape of output: {x.shape}")
        return x
