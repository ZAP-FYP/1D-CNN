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

# Load X and y arrays from the saved files
X = np.load('../YOLOPv2-1D_Coordinates/train_data/2X1.npy')   #X1.npy
y = np.load('../YOLOPv2-1D_Coordinates/train_data/2y1.npy')   #y1.npy
last_frame_of_x = np.load('../YOLOPv2-1D_Coordinates/train_data/2lastframe.npy')   #y.npy
X1 = np.load('../YOLOPv2-1D_Coordinates/train_data/2X.npy')   #X.npy
y1 = np.load('../YOLOPv2-1D_Coordinates/train_data/2y.npy')   #y.npy

shape_X = X1.shape
shape_y = y1.shape

print(f"Shape of X: {shape_X}")
print(f"Shape of y: {shape_y}")

flatten_y = y1.reshape((len(y1), -1))

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
            nn.Conv1d(in_channels=in_channels, out_channels=120, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=120, out_channels=60, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),
        )

        self.flatten = nn.Flatten()

        k = in_seq_len
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv1d):
                k = (k + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1

        # k = (in_seq_len // 2) // 5
        self.fc_layers = nn.Sequential(
            nn.Linear(60*k, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 500),
            
        )
        # k = (in_seq_len // 2) // 5
        # self.fc_layers = nn.Sequential(
        #     nn.Linear(60*k, 1000),
        #     nn.ReLU(),
        #     nn.Linear(1000, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 500),
            
        # )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x


test_flag = config('TEST_FLAG', cast=bool)
train_flag = config('TRAIN_FLAG', cast=bool)

start_time = time.time()

num_epochs = 15
batch_size = 5
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

count, in_channels, in_seq_len = X1.shape

if not test_flag:
    idx = int(count)
else:
    idx = int(count * 0.80)

train_dataset = VideoDataset(X1[:idx], flatten_y[:idx])
test_dataset = VideoDataset(X1[idx:], flatten_y[idx:])


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
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
    # current_epoch = checkpoint['epoch']
# model.train()
if train_flag:
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

            if not (i + 1) % 200:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

            # Save a checkpoint periodically
            if total_steps % 5000 == 0:
                save_checkpoint(epoch, model, optimizer, checkpoint_file)

        # # If you want to pause after the first set of data
        # if epoch == num_epochs // 2:
        #     break

# Save the final model checkpoint
    save_checkpoint(num_epochs, model, optimizer, checkpoint_file)

if test_flag:
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

    merged_predicted_y = torch.cat(predicted_y, dim=0)
    # Convert the merged tensor to a NumPy array
    merged_predicted_y_numpy = merged_predicted_y.numpy()
    np.save(f'merged_predicted_y_numpy.npy', merged_predicted_y_numpy)
    # last_test_x_frames = last_frame_of_x[idx:]

    # print("final_x_last ",last_test_x_frames.shape)
    # print("final_y ",merged_predicted_y_numpy.shape)

    # reconstructed_frames = np.zeros_like(merged_predicted_y_numpy)

    # # Loop through each data point
    # for data_point_index in range(merged_predicted_y_numpy.shape[0]):
    #     cumulative_frame = last_test_x_frames[data_point_index]
    #     # Loop through each frame difference in the data point
    #     for frame_diff_index in range(merged_predicted_y_numpy.shape[1]):
    #         frame_diff = merged_predicted_y_numpy[data_point_index, frame_diff_index]
    #         cumulative_frame = cumulative_frame + frame_diff
    #         reconstructed_frames[data_point_index, frame_diff_index] = cumulative_frame

    # print("reconstructed_frames ",reconstructed_frames.shape)
    # np.save(f'reconstructed_frames.npy', reconstructed_frames)
    # y1_test = y1[idx:]
    # np.save(f'y1_test.npy', y1_test)
    # print("y1_test ",y1_test.shape)
    mse = se / samples_count
    print(f"MSE of test data: {mse:.3f}")
