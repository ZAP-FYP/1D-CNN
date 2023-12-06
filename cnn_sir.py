import sys
import os
import numpy as np
#import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from decouple import config
from datetime import datetime
from torchsummary import summary



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

f = open('model/log.txt', 'w')
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


class ConvNet(nn.Module):
    
    def __init__(self, in_channels, in_seq_len):
          #super(ConvNet, self).__init__()
          #self.dropout = nn.Dropout(0.2)
          #self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=future_f, kernel_size=5, stride=1, padding=2)
          #self.conv2 = nn.Conv1d(in_channels=future_f, out_channels=future_f, kernel_size=5, stride=1, padding=2)
          #self.conv2 = nn.Conv1d(in_channels=5, out_channels=5, kernel_size=3, stride=1, padding=1)
        super(ConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=5, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=5, out_channels=future_f, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            #nn.Conv1d(in_channels=120, out_channels=5, kernel_size=5, stride=3, padding=2),
            #nn.ReLU()
        )

        #self.globalAvg = nn.AdaptiveAvgPool1d(10)

        self.fc_layers = nn.Sequential(
            nn.Linear(5*100, 500)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        #x=self.conv1(x)
        #x=F.relu(x)
        #x=self.conv2(x)
        #x=F.relu(x)
        #x= self.dropout(x)
        #x=self.conv2(x)
        #x=F.relu(x)
        x = torch.flatten(x, 2)
        #x = self.fc_layers(x)
        #x = self.globalAvg(x)
        #x = x.view(x.size(0), -1)
        #print(f"Shape of output: {x.shape}")
        
        return x
    
    
test_flag = config('TEST_FLAG', cast=bool)
train_flag = config('TRAIN_FLAG', cast=bool)
full_data_flag = config('FULL_DATA_FLAG', cast=bool)
future_f=config('FUTURE_FRAMES', cast=int)  #No of future frames to predict
start_f=config('START_FUTURE', cast=int)    #Startinf future frame


X_files = []
y_files = []
# Specify the directory path
directory_path = '../YOLOPv2-1D_Coordinates/train_data'

if full_data_flag:

    # Get a list of filenames in the directory
    filenames = os.listdir(directory_path)
    # Filter out directories, if needed
    folders = [filename for filename in filenames if not os.path.isfile(os.path.join(directory_path, filename))]
    # print(folders)

    for folder in folders:
        filenames = os.listdir(directory_path+"/"+folder)
        # print(filenames)
        for file in filenames:
            file = directory_path+"/"+folder+"/"+file
            if file[-5:] == "X.npy":
                # print("x file",file)
                X_file = np.load(file)
                X_files.append(X_file)
            elif file[-5:] == "y.npy":
                # print("yfile",file)

                y_file = np.load(file)
                y_files.append(y_file)
else:
    X_files.append(np.load(directory_path+"/20221124/1,2X.npy"))
    y_files.append(np.load(directory_path+"/20221124/1,2y.npy"))



X = np.vstack(X_files)  
y = np.vstack(y_files)  
# np.save(f'FullX.npy', X)
# np.save(f'Fully.npy', y)
y=y[:,start_f:(start_f+future_f),:]
shape_X = X.shape
shape_y = y.shape

print(f"Shape of X: {shape_X}")
print(f"Shape of y: {shape_y}")

flatten_y = y.reshape((len(y), -1))

num_epochs = 50
batch_size = 512
learning_rate = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

count, in_channels, in_seq_len = X.shape
if not test_flag:
    idx = int(count)
else:
    idx = int(count * 0.80)
val_idx = int(idx* 0.80)

DRR = 20#Data reduction ratio
#train_dataset = VideoDataset(X[:val_idx:5], y[:val_idx:5])
train_dataset = VideoDataset(X[::DRR], y[::DRR]) #32000
validation_dataset = VideoDataset(X[val_idx::DRR], y[val_idx::DRR])

test_dataset = VideoDataset(X[idx:], y[idx:])


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


print(f"Len of train_dataset X: {len(train_dataset)}")
print(f"Len of validation_dataset y: {len(validation_dataset)}")

print(f"Len of test_dataset y: {len(test_dataset)}")
print(f"in_channels: {in_channels}")

model = ConvNet(in_channels, in_seq_len).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

checkpoint_file = 'model/model_checkpoint.pth'
if False: #os.path.isfile(checkpoint_file):
    print("Loading saved model...")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    current_epoch = checkpoint['epoch']
print(f"Model summary: {summary(model, (in_channels, in_seq_len))}")

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

            if not (i + 1) % int(batch_size/4):
                print(f'Epoch [{epoch+1}/{num_epochs}] Batch[{i}] Current time:{datetime.now()}')

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

            val_loss /= len(validation_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f} Validation Loss: {val_loss:.4f}')

        # Check if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            consecutive_no_improvement = 0
            save_checkpoint(epoch, model, optimizer, "model/best_model_checkpoint.pth")
        else:
            consecutive_no_improvement += 1

        # Check for early stopping
        if consecutive_no_improvement >= patience:
            print(f'best_val_loss {best_val_loss}')
            print(f'Early stopping at epoch {epoch+1}')
            break
        print(f'best_val_loss {best_val_loss}')

    #Save the final model checkpoint
    save_checkpoint(num_epochs, model, optimizer, checkpoint_file)
    
if test_flag:
    model.eval()
    se = 0
    samples_count = 0


    # Assuming y_hat is your tensor with the same shape [5, 5, 125]

    # Create a directory to save the visualizations
    output_folder = "visualizations"
    os.makedirs(output_folder, exist_ok=True)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            y_hat = model(images)
            print(y_hat.shape)
            loss = criterion(y_hat, labels)

            se += (loss.item() * labels.size(0))
            samples_count += labels.size(0)

            # Reshape labels and y_hat
            images = images.view(labels.size(0), 10, 100)
            labels = labels.view(labels.size(0), future_f, 100)
            y_hat = y_hat.view(y_hat.size(0), future_f, 100)

            for i in range(labels.size(0)):  # Loop through each sample in the batch
                image_frame = images[i]  # Get the input for this sample
                label_frame = labels[i]  # Get the label frame for this sample
                y_hat_frame = y_hat[i]  # Get the corresponding y_hat frame

                # Create subfolders for each sample
                sample_folder = os.path.join(output_folder, f"sample_{i}")
                os.makedirs(sample_folder, exist_ok=True)

                # Visualize and save each label and y_hat frame
                for j in range(future_f):  # Loop through each frame in the sample
                    label_array = label_frame[j].cpu().detach().numpy()  # Convert to NumPy array
                    y_hat_array = y_hat_frame[j].cpu().detach().numpy()  # Convert to NumPy array
                    

                    # Plot both label and y_hat arrays in the same figure
                    plt.figure(figsize=(8, 4))
                    plt.plot(label_array, label="Label Array")
                    plt.plot(y_hat_array, label="y_hat Array")
                    plt.title(f"Output Frame {j}")
                    plt.legend()  # Add a legend to differentiate between Label Array and y_hat Array

                    # Save the figure
                    plt.savefig(os.path.join(sample_folder, f"sample_{i}_frame_{j}.png"))
                    plt.close()

                for j in range(10):  # Loop through each frame in the sample
                    
                    image_array = image_frame[j].cpu().detach().numpy()  # Convert to NumPy array

                    # Plot both label and y_hat arrays in the same figure
                    plt.figure(figsize=(8, 4))
                    plt.plot(image_array, label="Image Array")
                    
                    plt.title(f"Input Frame {j}")
                    plt.legend()  # Add a legend to differentiate between Label Array and y_hat Array

                    # Save the figure
                    plt.savefig(os.path.join(sample_folder, f"input_{i}_frame_{j}.png"))
                    plt.close()

    mse = se / samples_count
    print(f"MSE of test data: {mse:.3f}")

if False:
    output_folder = "Input_visualizations"
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)


            #se += (loss.item() * labels.size(0))
            #samples_count += labels.size(0)

            # Reshape labels and y_hat
            labels = labels.view(labels.size(0), future_f, 100)
            images = images.view(images.size(0), 10, 100)

            for i in range(labels.size(0)):  # Loop through each sample in the batch
                label_frame = labels[i]  # Get the label frame for this sample
                images_frame = images[i]  # Get the corresponding input frame

                # Create subfolders for each sample
                sample_folder = os.path.join(output_folder, f"sample_{i}")
                os.makedirs(sample_folder, exist_ok=True)


                # Visualize and save each training frame
                for j in range(10):  # Loop through each frame in the sample
                    image_array = images_frame[j].cpu().detach().numpy()  # Convert to NumPy array

                    # Plot both label and y_hat arrays in the same figure
                    plt.figure(figsize=(8, 4))
                    plt.plot(image_array, label="Input Array")
                    plt.title(f"Input Frame {j}")

                    # Save the figure
                    plt.savefig(os.path.join(sample_folder, f"I_sample_{i}_frame_{j}.png"))
                    plt.close()
                    
                # Visualize and save each label frame
                for j in range(future_f):  # Loop through each frame in the sample
                    label_array = label_frame[j].cpu().detach().numpy()  # Convert to NumPy array

                    # Plot both label and y_hat arrays in the same figure
                    plt.figure(figsize=(8, 4))
                    plt.plot(label_array, label="Label Array")
                    plt.title(f"Label Frame {j}")

                    # Save the figure
                    plt.savefig(os.path.join(sample_folder, f"L_sample_{i}_frame_{j}.png"))
                    plt.close()
#os.chdir('D:\\UoM\\Research\\Drivable Area\\1D-CNN')
#os.getcwd()