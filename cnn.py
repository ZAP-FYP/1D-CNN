import numpy as np
import cv2
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from decouple import config



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
        super(ConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=120, kernel_size=5, stride=3, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=120, out_channels=60, kernel_size=5, stride=3, padding=2),
            nn.ReLU(),
            
        )

        self.flatten = nn.Flatten()

        # Calculate the output size after convolution without pooling
        k = in_seq_len
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv1d) :
                k = (k + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
        print(f"k {k}")
        self.fc_layers = nn.Sequential(
            nn.Linear(60*k, 1000),
            nn.ReLU(),
            # nn.Dropout(0.5),  # Add dropout for regularization
            nn.Linear(1000, 100),
            nn.ReLU(),
            # nn.Dropout(0.5), 
            nn.Linear(100, 500),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x
    
test_flag = config('TEST_FLAG', cast=bool)
train_flag = config('TRAIN_FLAG', cast=bool)

X_files = []
y_files = []

# Specify the directory path
directory_path = '../YOLOPv2-1D_Coordinates/train_data'
# Get a list of filenames in the directory
filenames = os.listdir(directory_path)
# Filter out directories, if needed
folders = [filename for filename in filenames if not os.path.isfile(os.path.join(directory_path, filename))]
print(folders)

for folder in folders:
    filenames = os.listdir(directory_path+"/"+folder)
    print(filenames)
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



X = np.vstack(X_files)  
y = np.vstack(y_files)  
np.save(f'FullX.npy', X)
np.save(f'Fully.npy', y)
shape_X = X.shape
shape_y = y.shape

print(f"Shape of X: {shape_X}")
print(f"Shape of y: {shape_y}")

flatten_y = y.reshape((len(y), -1))

num_epochs = 50
batch_size = 5
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

count, in_channels, in_seq_len = X.shape
if not test_flag:
    idx = int(count)
else:
    idx = int(count * 0.80)
val_idx = int(idx* 0.80)

train_dataset = VideoDataset(X[:val_idx], flatten_y[:val_idx])
validation_dataset = VideoDataset(X[val_idx:], flatten_y[val_idx:])

test_dataset = VideoDataset(X[idx:], flatten_y[idx:])


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


print(f"Len of train_dataset X: {len(train_dataset)}")
print(f"Len of test_dataset y: {len(test_dataset)}")

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
if os.path.isfile(checkpoint_file):
    print("Loading saved model...")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    current_epoch = checkpoint['epoch']

if train_flag:
    # Define early stopping parameters
    patience = 5  # Number of consecutive epochs without improvement
    best_val_loss = float('inf')
    consecutive_no_improvement = 0
    for epoch in range(current_epoch, num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            y_hat = model(images)
            loss = criterion(y_hat, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_steps += 1

            # if not (i + 1) % 200:
                # print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

            # Save a checkpoint periodically
            if total_steps % 5000 == 0:
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

            val_loss /= len(test_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

        # Check if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            consecutive_no_improvement = 0
            save_checkpoint(epoch, model, optimizer, checkpoint_file)
        else:
            consecutive_no_improvement += 1

        # Check for early stopping
        if consecutive_no_improvement >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
# Save the final model checkpoint
    # save_checkpoint(num_epochs, model, optimizer, checkpoint_file)
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
            labels = labels.view(labels.size(0), 5, 100)
            y_hat = y_hat.view(y_hat.size(0), 5, 100)

            for i in range(labels.size(0)):  # Loop through each sample in the batch
                label_frame = labels[i]  # Get the label frame for this sample
                y_hat_frame = y_hat[i]  # Get the corresponding y_hat frame

                # Create subfolders for each sample
                sample_folder = os.path.join(output_folder, f"sample_{i}")
                os.makedirs(sample_folder, exist_ok=True)

                # Visualize and save each label and y_hat frame
                for j in range(5):  # Loop through each frame in the sample
                    label_array = label_frame[j].cpu().detach().numpy()  # Convert to NumPy array
                    y_hat_array = y_hat_frame[j].cpu().detach().numpy()  # Convert to NumPy array

                    # Plot both label and y_hat arrays in the same figure
                    plt.figure(figsize=(8, 4))
                    plt.plot(label_array, label="Label Array")
                    plt.plot(y_hat_array, label="y_hat Array")
                    plt.title(f"Frame {j}")
                    plt.legend()  # Add a legend to differentiate between Label Array and y_hat Array

                    # Save the figure
                    plt.savefig(os.path.join(sample_folder, f"sample_{i}_frame_{j}.png"))
                    plt.close()

    mse = se / samples_count
    print(f"MSE of test data: {mse:.3f}")
