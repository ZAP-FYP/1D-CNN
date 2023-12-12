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

# file_path = 'frame_coords_IMG_0260.npy'
# num_samples = 200
# frame_length = 100
# num_channels = 10
# # Load the entire array from the file
# data_array = np.load(file_path)

# single_frame = torch.from_numpy(data_array[0])

# X = single_frame.expand((num_samples, num_channels, frame_length))
# y = single_frame.expand((num_samples, 5, frame_length))  # Adjusted frame length


# Function to generate a frame with a given velocity
def generate_frame(base_frame, velocity):
    return base_frame + velocity

# Generate a simple base frame (e.g., a rectangle)
frame_length = 100
base_frame = np.zeros(frame_length)
rectangle_width = 20
base_frame[40:40 + rectangle_width] = 1

# Define a simple velocity (e.g., constant increase)
velocity = 2

# Generate 15 frames with incremental changes
frames = [generate_frame(base_frame, i * velocity) for i in range(15)]

# Separate frames into X and y
X = np.array(frames[:10])  # First 10 frames as 10 channels
y = np.array(frames[10:])  # Last 5 frames as 5 channels

# Duplicate the data sample 200 times
X = np.tile(X, (200, 1, 1))
y = np.tile(y, (200, 1, 1))

# X = X.numpy()
# y = y.numpy()
# Print the shapes
print("Shape of input_data:", X.shape)
print("Shape of output_labels:", y.shape)


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


import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, in_channels, in_seq_len):
        super(ConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=5, kernel_size=3, stride=1, padding=1),  # Adjusted padding to make the output size the same
            nn.ReLU()
        )
        # self.global_avg_pooling = nn.AdaptiveAvgPool1d(500)  # Output size will be (batch_size, out_channels, 1)

        conv_out_size = in_seq_len // 1  
        self.fc_layers = nn.Sequential(
            nn.Linear(5 * conv_out_size, 500)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.fc_layers(x)
        # x = self.global_avg_pooling(x)
        return x



num_epochs = 50
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
predicted_y = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        y_hat = model(images)
        # print(y_hat.shape)
        loss = criterion(y_hat, labels)

        se += (loss.item() * labels.size(0))
        samples_count += labels.size(0)

        # Reshape labels and y_hat
        labels = labels.view(labels.size(0), 5, 100)
        y_hat = y_hat.view(y_hat.size(0), 5, 100)
        predicted_y.append(y_hat)
        print(y_hat.shape)

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

