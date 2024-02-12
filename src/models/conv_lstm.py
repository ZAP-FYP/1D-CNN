import torch
import torch.nn as nn

# class ConvLSTM1D(nn.Module):
#     def __init__(self, input_size, hidden_size, kernel_size, num_layers):
#         super(ConvLSTM1D, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.kernel_size = kernel_size
#         self.num_layers = num_layers

#         # Convolutional LSTM layers
#         self.conv_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

#         # Fully connected layer to map the LSTM output to the desired output size
#         self.fc = nn.Linear(hidden_size, 500)

#     def forward(self, x):
#         # Input shape: (batch_size, sequence_length, input_size)

#         # print("Input Shape:", x.shape)

#         # Initialize hidden and cell states
#         batch_size, _, _ = x.size()
#         h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

#         # ConvLSTM forward pass
#         lstm_out, _ = self.conv_lstm(x, (h0, c0))

#         # print("LSTM Output Shape:", lstm_out.shape)

#         # Take the output of the last time step
#         lstm_last_output = lstm_out[:, -1, :]

#         # Fully connected layer
#         output = self.fc(lstm_last_output)

#         return output


class ConvLSTM1D(nn.Module):
    def __init__(
        self, input_size, hidden_size, kernel_size, num_layers, bidirectional=False
    ):
        super(ConvLSTM1D, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Convolutional LSTM layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=10, out_channels=1, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
        )
        self.conv_lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Adjust the size of the fully connected layer output accordingly
        fc_input_size = 2 * hidden_size if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, 500)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)

        # print("Input Shape:", x.shape)

        # Initialize hidden and cell states
        batch_size, _, _ = x.size()
        # Initialize hidden and cell states
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(
            self.num_layers * num_directions, batch_size, self.hidden_size
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers * num_directions, batch_size, self.hidden_size
        ).to(x.device)
        # h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        x = self.conv_layers(x)

        # ConvLSTM forward pass
        lstm_out, _ = self.conv_lstm(x, (h0, c0))

        # print("LSTM Output Shape:", lstm_out.shape)

        # Take the output of the last time step
        lstm_last_output = lstm_out[:, -1, :]

        # Fully connected layer
        output = self.fc(lstm_last_output)

        # print("Output Shape:", output.shape)

        return output
