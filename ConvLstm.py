import torch
import torch.nn as nn
import numpy as np
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from decouple import config
from datetime import datetime
from torchsummary import summary
from data_creator import get_X_y

import sys
model_name = "ConvLSTM/LSTM-12"



checkpoint_file = 'model/ConvLSTM/'+model_name+'/model_checkpoint.pth'
if not os.path.exists("model/"+model_name):
    os.makedirs("model/"+model_name)
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

f = open('model/'+model_name+'/log.txt', 'w')
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)

class VideoDataset(Dataset):
    def __init__(self, X, y):
        self.x = torch.from_numpy(X).float().requires_grad_()
        self.y = torch.from_numpy(y).float().requires_grad_()
        self.n_samples = y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

class ConvLSTM1D(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, num_layers, bidirectional=True):
        super(ConvLSTM1D, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Convolutional LSTM layers
        self.conv_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)

                # Adjust the size of the fully connected layer output accordingly
        fc_input_size = 2 * hidden_size if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, input_size)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)

        # print("Input Shape:", x.shape)

        # Initialize hidden and cell states
        batch_size, _, _ = x.size()
        # Initialize hidden and cell states
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(x.device)
        # h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # ConvLSTM forward pass
        lstm_out, _ = self.conv_lstm(x, (h0, c0))

        # print("LSTM Output Shape:", lstm_out.shape)

        # Take the output of the last time step
        lstm_last_output = lstm_out[:, -1, :]

        # Fully connected layer
        output = self.fc(lstm_last_output)

        # print("Output Shape:", output.shape)

        return output

num_epochs = 1000
batch_size = 25
learning_rate = 0.001
input_size = 100
hidden_size = 50
kernel_size = 3
num_layers = 3
model = ConvLSTM1D(input_size, hidden_size, kernel_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


test_flag = config('TEST_FLAG', cast=bool)
train_flag = config('TRAIN_FLAG', cast=bool)
full_data_flag = config('FULL_DATA_FLAG', cast=bool)
prev_f = config('PREV_FRAMES', cast=int)
future_f=config('FUTURE_FRAMES', cast=int)  #No of future frames to predict
start_f=config('START_FUTURE', cast=int)    #Startinf future frame
DRR = config('DATA_REDUCTION_RATE', cast=int)


X, y = get_X_y(prev_f, future_f)

X = np.array(X)
y = np.array(y)


shape_X = X.shape
shape_y = y.shape

print(f"Shape of X: {shape_X}")
print(f"Shape of y: {shape_y}")

flatten_y = y.reshape((len(y), -1))



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

count, in_channels, in_seq_len = X.shape
if not test_flag:
    idx = int(count)
else:
    idx = int(count * 0.80)
val_idx = int(idx* 0.80)

# train_dataset = VideoDataset(X[:val_idx:DRR], flatten_y[:val_idx:DRR])
# validation_dataset = VideoDataset(X[val_idx::DRR], flatten_y[val_idx::DRR])
train_dataset = VideoDataset(X[::DRR], y[::DRR]) #32000
validation_dataset = VideoDataset(X[val_idx::DRR], y[val_idx::DRR])

test_dataset = VideoDataset(X[idx:], flatten_y[idx:])


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


print(f"Len of train_dataset X: {len(train_dataset)}")
print(f"Len of validation_dataset y: {len(validation_dataset)}")

print(f"Len of test_dataset y: {len(test_dataset)}")
print(f"in_channels: {in_channels}")

current_epoch = 0
total_steps = 0



def save_checkpoint(epoch, model, optimizer, filename):
    print("Saving model checkpoint...")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


if os.path.isfile(checkpoint_file):
    print("Loading saved model...")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    current_epoch = checkpoint['epoch']
# print(f"Model summary : {summary(model, (in_channels, in_seq_len))}")
if train_flag:
    # Define early stopping parameters
    print("Starting training...")
    patience = 5  # Number of consecutive epochs without improvement
    best_val_loss = float('inf')
    consecutive_no_improvement = 0
    for epoch in range(current_epoch, num_epochs):
        train_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # print(f"images.size() {images.size()}")

            y_hat = model(images)
            # Reshape y_hat to match the size of labels along the second dimension
            # y_hat_reshaped = y_hat.view(labels.size(0), labels.size(1), -1)
            # print(f"Size of y_hat: {y_hat.size()}")
            # print(f"Size of y_hat_reshaped: {y_hat_reshaped.size()}")
            # print(f"Size of labels: {labels.size()}")   
            loss = criterion(y_hat, labels)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_steps += 1

            if not (i + 1) % 400:
                print(f'Epoch [{epoch+1}/{num_epochs}] Current time:{datetime.now()}')

        train_loss /= len(train_loader)

        # save_checkpoint(epoch, model, optimizer, checkpoint_file)

        # Validate the model at the end of each epoch
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            for i, (val_images, val_labels) in enumerate(validation_loader):
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_labels).item()
                            # Reshape labels and y_hat
            labels = val_labels.view(val_labels.size(0), 1, 100)
            y_hat = val_outputs.view(val_outputs.size(0), 1, 100)

                # for i in range(labels.size(0)):  # Loop through each sample in the batch
            label_frame = labels[0]  # Get the label frame for this sample
            y_hat_frame = y_hat[0]  # Get the corresponding y_hat frame

            # Create subfolders for each sample
            output_folder = "visualizations/validation/"+model_name
            sample_folder = os.path.join(output_folder, f"sample_{0}")
            os.makedirs(sample_folder, exist_ok=True)

            # Visualize and save each label and y_hat frame
            # for j in range(5):  # Loop through each frame in the sample
            label_array = label_frame.cpu().detach().numpy()  # Convert to NumPy array
            y_hat_array = y_hat_frame.cpu().detach().numpy()  # Convert to NumPy array

            # Plot both label and y_hat arrays in the same figure
            # plt.figure(figsize=(8, 4))
            # plt.plot(label_array, label="Label Array")
            # plt.plot(y_hat_array, label="y_hat Array")
            plt.figure(figsize=(8, 4))
            plt.plot(label_array[0], label="Label Array")
            plt.plot(y_hat_array[0], label="y_hat Array")
            plt.title(f"Frame {future_f}")
            plt.legend()  # Add a legend to differentiate between Label Array and y_hat Array

            # Save the figure
            plt.savefig(os.path.join(sample_folder, f"sample_{i}_frame_{future_f}.png"))
            plt.close()

            val_loss /= len(validation_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f} Validation Loss: {val_loss:.4f}')

        # Check if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            consecutive_no_improvement = 0
            save_checkpoint(epoch, model, optimizer, 'model/'+model_name+'/best_model_checkpoint.pth')
        else:
            consecutive_no_improvement += 1

        # Check for early stopping
        if consecutive_no_improvement >= patience:
            print(f'best_val_loss {best_val_loss}')
            print(f'Early stopping at epoch {epoch+1}')
            break
        print(f'best_val_loss {best_val_loss}')
