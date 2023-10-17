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
import time

# Load X and y arrays from the saved files
# X = np.load('X.npy')   #X.npy
# y = np.load('y.npy')   #y.npy

# X = np.load('train_data/IMG_0260.MOVX.npy')   
# y = np.load('train_data/IMG_0260.MOVy.npy') 

X_files = ['train_data/IMG_0260.MOVX.npy', 'train_data/IMG_0261.MOVX.npy']
y_files = ['train_data/IMG_0260.MOVy.npy', 'train_data/IMG_0261.MOVy.npy']

X_arrays = []
y_arrays = []

for file_path in X_files:
    X_array = np.load(file_path)
    X_arrays.append(X_array)
X = np.vstack(X_arrays)  

for file_path in y_files:
    y_array = np.load(file_path)
    y_arrays.append(y_array)
y = np.vstack(y_arrays)  

shape_X = X.shape
shape_y = y.shape

print(f"Shape of merged X: {shape_X}")
print(f"Shape of merged y: {shape_y}")

flatten_y = y.reshape((len(y), -1))

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
            nn.Conv1d(in_channels=in_channels, out_channels=120, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # nn.Conv1d(in_channels=120, out_channels=240, kernel_size=3, stride=1, padding=2),
            # nn.ReLU(),
            nn.Conv1d(in_channels=120, out_channels=60, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),
        )

        self.flatten = nn.Flatten()

        # k = in_seq_len
        # for layer in self.conv_layers:
        #     if isinstance(layer, nn.Conv1d):
        #         k = (k + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1

        k = (in_seq_len // 2) // 5
        self.fc_layers = nn.Sequential(
            nn.Linear(60*k, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 500),
            
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x



start_time = time.time()

num_epochs = 15
batch_size = 5
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

count, in_channels, in_seq_len = X.shape
idx = int(count * 0.80)

train_dataset = VideoDataset(X[:idx], flatten_y[:idx])
test_dataset = VideoDataset(X[idx:], flatten_y[idx:])


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


print(f"Len of train_dataset X: {len(train_dataset)}")
print(f"Len of test_dataset y: {len(test_dataset)}")

model = ConvNet(in_channels, in_seq_len).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# model.train()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # images.requires_grad = True
        # labels.requires_grad = True

        y_hat = model(images)

        j = criterion(y_hat, labels)

        if not (i + 1) % 2000:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{len(train_loader)}, training error {j.item():.3f}')

        optimizer.zero_grad()
        # j.requires_grad = True
        j.backward()

        optimizer.step()

model.eval()
se = 0
samples_count = 0


import os
import matplotlib.pyplot as plt

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

end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")