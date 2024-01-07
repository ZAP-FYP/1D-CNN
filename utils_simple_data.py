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
from data_creator import generate_base_frame,generate_moved_frame,visualize_frames,get_X_y
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
    # viz_labels_shape = (viz_labels.size(0), 100, 5)
    # reshaped_viz_labels = viz_labels.reshape(viz_labels_shape)
    # labels = np.transpose(reshaped_viz_labels, (0, 2, 1))

    # viz_outputs_shape = (viz_outputs.size(0), 100, 5)
    # reshaped_viz_outputs = viz_outputs.reshape(viz_outputs_shape)
    # y_hat = np.transpose(reshaped_viz_outputs, (0, 2, 1))

    labels = viz_labels.view(viz_labels.size(0), 5, 100)
    y_hat = viz_outputs.view(viz_outputs.size(0), 5, 100)

    if n_th_frame:
        outer_loop = 1
        inner_loop = 1
    else:
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

# Set the length of the frame and the width of the car
frame_length = 100
car_width = 20

# Generate the base frame with a car
base_frame = generate_base_frame(frame_length, car_width)

# Initialize parameters for car movement
min_velocity = 3
max_velocity = 10

# Generate frames with mixed movements
num_frames = 60
frames = [base_frame]
# Define the pattern
acceleration_frames = 60
constant_velocity_frames = 0
deceleration_frames = 0

# Create the pattern for the entire sequence
pattern = np.concatenate([
    np.linspace(min_velocity, max_velocity, acceleration_frames),
    np.full(constant_velocity_frames, max_velocity),
    np.linspace(max_velocity, min_velocity, deceleration_frames)
])

# Repeat the pattern to cover the entire sequence
velocity_sequence = np.tile(pattern, num_frames // len(pattern))

# Generate frames based on the velocity sequence
for i in range(num_frames):
    # horizontal_velocity = np.random.randint(-max_velocity, max_velocity + 1)
    horizontal_velocity = 0
    vertical_velocity = velocity_sequence[i]
    frame = generate_moved_frame(frames[-1], horizontal_velocity, vertical_velocity)
    frames.append(frame)

save_directory = "frames_plots"
os.makedirs(save_directory, exist_ok=True)
save_filename="frames_plot.png"
save_path = os.path.join(save_directory, save_filename)
visualize_frames(frames,save_path=save_path)

# Creating x and y sequences
window_x = 10
window_y = 5

# Split frames into train and test sets
train_frames = frames[:20]  # Adjust as needed
val_frames = frames[20:40]
test_frames = frames[40:]

x_train = np.array([train_frames[i:i + window_x] for i in range(len(train_frames) - window_x - window_y)])
y_train = np.array([train_frames[i + window_x:i + window_x + window_y] for i in range(len(train_frames) - window_x - window_y)])

x_val = np.array([val_frames[i:i + window_x] for i in range(len(val_frames) - window_x - window_y)])
y_val = np.array([val_frames[i + window_x:i + window_x + window_y] for i in range(len(val_frames) - window_x - window_y)])

x_test = np.array([test_frames[i:i + window_x] for i in range(len(test_frames) - window_x - window_y)])
y_test = np.array([test_frames[i + window_x:i + window_x + window_y] for i in range(len(test_frames) - window_x - window_y)])

# Duplicate the training set
num_duplicates = 200
x_train_duplicate = np.tile(x_train, (num_duplicates, 1, 1))
y_train_duplicate = np.tile(y_train, (num_duplicates, 1, 1))

# Print the shapes of the NumPy arrays
print("X_train_duplicate_shape:", x_train_duplicate.shape)
print("Y_train_duplicate_shape:", y_train_duplicate.shape)
print("X_test_shape:", x_test.shape)
print("Y_test_shape:", y_test.shape)
print("X_val_shape:", x_val.shape)
print("Y_val_shape:", y_val.shape)

flatten_y = y_train_duplicate.reshape((len(y_train_duplicate), -1))
flatten_y_val = y_val.reshape((len(y_val), -1))
flatten_y_test = y_test.reshape((len(y_test), -1))

# transposed_y = np.transpose(y_train_duplicate, (0, 2, 1))
# # print(transposed_y.shape)
# flatten_y_channel_wise = transposed_y.reshape((100, -1))
# # print(flatten_y_channel_wise.shape)

# transposed_y_test = np.transpose(y_test, (0, 2, 1))
# flatten_y_test_channel_wise = transposed_y_test.reshape(6, -1)

# transposed_y_val = np.transpose(y_val, (0, 2, 1))
# flatten_y_val_channel_wise = transposed_y_val.reshape((5, -1))

count, in_channels, in_seq_len = x_train_duplicate.shape

# train_dataset = VideoDataset(x_train_duplicate, flatten_y_channel_wise) 
# validation_dataset = VideoDataset(x_val, flatten_y_val_channel_wise)
# test_dataset = VideoDataset(x_test, flatten_y_test_channel_wise)

train_dataset = VideoDataset(x_train_duplicate, flatten_y) 
validation_dataset = VideoDataset(x_val, flatten_y_val)
test_dataset = VideoDataset(x_test, flatten_y_test)

print(f"Len of train_dataset X: {len(train_dataset)}")
print(f"Len of validation_dataset y: {len(validation_dataset)}")
print(f"Len of test_dataset y: {len(test_dataset)}")


num_epochs = 1000
batch_size = 5
learning_rate = 0.001

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    patience = 15  # Number of consecutive epochs without improvement
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
