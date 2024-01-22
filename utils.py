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
from data_creator import get_X_y, create_averaged_frames
from cnn import ConvNet
from CnvLstm import ConvLSTM1D
import sys


test_flag = config('TEST_FLAG', cast=bool)
train_flag = config('TRAIN_FLAG', cast=bool)
full_data_flag = config('FULL_DATA_FLAG', cast=bool)
n_th_frame = config('N_TH_FRAME', cast=bool)
prev_f = config('PREV_FRAMES', cast=int)
future_f=config('FUTURE_FRAMES', cast=int)  #No of future frames to predict
start_f=config('START_FUTURE', cast=int)    #Startinf future frame
DRR = config('DATA_REDUCTION_RATE', cast=int)
model_name = config('MODEL_NAME')

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


class VideoDataset(Dataset):
    def __init__(self, X, y):
        self.x = torch.from_numpy(X).float().requires_grad_()
        self.y = torch.from_numpy(y).float().requires_grad_()
        self.n_samples = y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


def save_checkpoint(epoch, model, optimizer, filename):
    print("Saving model checkpoint...")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def visualize(viz_labels, viz_outputs, output_folder):

    if n_th_frame:
        labels = viz_labels.view(viz_labels.size(0), 1, 100)
        y_hat = viz_outputs.view(viz_outputs.size(0), 1, 100)
        outer_loop = 1
        inner_loop = 1
    else:
        labels = viz_labels.view(viz_labels.size(0), future_f, 100)
        y_hat = viz_outputs.view(viz_outputs.size(0), future_f, 100)
        outer_loop = labels.size(0)
        inner_loop = future_f

    # Loop through each sample in the batch
    for i in range(outer_loop):
        label_frame = labels[i]  # Get the label frame for this sample
        y_hat_frame = y_hat[i]  # Get the corresponding y_hat frame

        # Create subfolders for each sample
        sample_folder = os.path.join(output_folder, f"sample_{i}")
        os.makedirs(sample_folder, exist_ok=True)

        # Visualize and save each label and y_hat frame
        for j in range(inner_loop):  # Loop through each frame in the sample
            label_array = label_frame.cpu().detach().numpy()  # Convert to NumPy array
            y_hat_array = y_hat_frame.cpu().detach().numpy()  # Convert to NumPy array

            plt.figure(figsize=(8, 4))
            plt.plot(label_array[j], label="Label Array")
            plt.plot(y_hat_array[j], label="y_hat Array")
            plt.title(f"Frame {j}")
            plt.legend()  # Add a legend to differentiate between Label Array and y_hat Array

            # Save the figure
            plt.savefig(os.path.join(sample_folder, f"sample_{i}_frame_{j}.png"))
            plt.close()




X, y = get_X_y(prev_f, future_f, n_th_frame)

X_avg, y_avg = create_averaged_frames(X, y, DRR)

flatten_y = y_avg.reshape((len(y_avg), -1))

count, in_channels, in_seq_len = X_avg.shape
if not test_flag:
    idx = int(count)
else:
    idx = int(count * 0.80)
val_idx = int(idx* 0.80)

if DRR != 0:
    # train_dataset = VideoDataset(X_avg[::DRR], flatten_y[::DRR]) 
    train_dataset = VideoDataset(X_avg, flatten_y) 
    # validation_dataset = VideoDataset(X_avg[val_idx::DRR], flatten_y[val_idx::DRR])
    validation_dataset = VideoDataset(X_avg[val_idx:], flatten_y[val_idx:])

else:
    train_dataset = VideoDataset(X[::], flatten_y[::]) 
    validation_dataset = VideoDataset(X[val_idx::], flatten_y[val_idx::])
test_dataset = VideoDataset(X[idx:], flatten_y[idx:])

print(f"Len of train_dataset X: {len(train_dataset)}")
print(f"Len of validation_dataset y: {len(validation_dataset)}")
print(f"Len of test_dataset y: {len(test_dataset)}")



num_epochs = 1000
batch_size = 25
learning_rate = 0.001

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# input_size = 100
# hidden_size = 50
# kernel_size = 3
# num_layers = 3
# model = ConvLSTM1D(input_size, hidden_size, kernel_size, num_layers)

model = ConvNet(in_channels, in_seq_len, 5).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


current_epoch = 0
total_steps = 0


checkpoint_file = 'model/'+model_name+'/model_checkpoint.pth'
if not os.path.exists("model/"+model_name):
    os.makedirs("model/"+model_name)

f = open('model/'+model_name+'/log.txt', 'w')
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)

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

            y_hat = model(images)
            loss = criterion(y_hat, labels)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_steps += 1

            if not (i + 1) % 400:
                print(f'Epoch [{epoch+1}/{num_epochs}] Current time:{datetime.now()}')

        train_loss /= len(train_loader)

        save_checkpoint(epoch, model, optimizer, checkpoint_file)

        # Validate the model at the end of each epoch
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            for i, (val_images, val_labels) in enumerate(validation_loader):
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_labels).item()

                output_folder = "visualizations/validation/"+model_name
                visualize(val_labels, val_outputs, output_folder)

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


if test_flag:
    model.eval()
    se = 0
    samples_count = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            y_hat = model(images)
            print(y_hat.shape)
            loss = criterion(y_hat, labels)

            se += (loss.item() * labels.size(0))
            samples_count += labels.size(0)

            output_folder = "visualizations/test/"+model_name
            visualize(labels, y_hat, output_folder)
            
    mse = se / samples_count
    print(f"MSE of test data: {mse:.3f}")
