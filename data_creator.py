import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

def visualize(x, y, output_folder):
    num_samples, num_frames_x, frame_length_x = x.shape
    _, num_frames_y, frame_length_y = y.shape 

    for sample_index in range(num_samples):
        sample_folder = os.path.join(output_folder, f"sample_{sample_index}")
        os.makedirs(sample_folder, exist_ok=True)

        plt.figure(figsize=(15, 4))

        # Plot x
        for frame_index in range(num_frames_x):
            x_frame = x[sample_index, frame_index]
            plt.plot(x_frame, label=f'Sample {sample_index}, Frame {frame_index} - Input (x)')

        # Plot y
        for frame_index in range(num_frames_y):
            y_frame = y[sample_index, frame_index]
            plt.plot(y_frame, label=f'Sample {sample_index}, Frame {frame_index} - Output (y)', color='red')

        plt.xlabel('Time Steps')
        plt.ylabel('Values')
        plt.legend()

        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(sample_folder, f"sample_{sample_index}_visualization.png"))
        plt.close()






def get_X_y(prev_frames, future_frames, n_th_frame):
    X_arr=[]
    y_arr=[]
    directory_path = '../YOLOPv2-1D_Coordinates/data_npy'

    # filenames = os.listdir(directory_path)
    filenames = [f for f in os.listdir(directory_path) if not f.startswith(".DS_Store")]

    print(filenames)

    for _file in filenames:
        file = directory_path+"/"+_file
        X_file = np.load(file)

        window_size = prev_frames + future_frames
        X = [X_file[i:i+prev_frames] for i in range(len(X_file[:-window_size]))]

        if n_th_frame:
            y = [X_file[i+prev_frames+future_frames] for i in range(len(X_file[:-window_size]))]
        else:
            y = [X_file[(i+prev_frames):(i+prev_frames+future_frames)] for i in range(len(X_file[:-window_size]))]

        X_arr.extend(X)
        y_arr.extend(y)
    
    X_arr = np.array(X_arr)
    y_arr = np.array(y_arr)

    shape_X = X_arr.shape
    shape_y = y_arr.shape
    print(f"Shape of X: {shape_X}")
    print(f"Shape of y: {shape_y}")


    # output_folder_example = "samples"

    # visualize(X_arr, y_arr, output_folder_example)
        
    return X_arr, y_arr


def get_equidistant_X_y(prev_frames, future_frames):
    X_files=[]
    y_files=[]
    directory_path = '../YOLOPv2-1D_Coordinates/data_npy'

    filenames = os.listdir(directory_path)
    print(filenames)

    for _file in filenames:
        file = directory_path+"/"+_file
        X_file = np.load(file)
        X_files.extend(X_file)

    window_size = prev_frames * future_frames

    # X = [X_files[i:i+prev_frames] for i in range(len(X_files[:-window_size]))]
    X,y = [], []
    for i in range(len(X_files[:-window_size-future_frames])):
        # if i % future_frames == 0:
        temp=[]
        for x in range(i,i+window_size,future_frames):
            temp.append(X_files[x])
            # X.append(X_files[x])
        # print(len(temp))
        X.append(temp)
        y.append(X_files[i+window_size+future_frames])
    print(len(temp))

    # y = [[X_files[i+prev_frames+future_frames-1]] for i in range(len(X_files[:-window_size]))]

    X = np.array(X)
    y = np.array(y)

    shape_X = X.shape
    shape_y = y.shape
    print(f"Shape of X: {shape_X}")
    print(f"Shape of y: {shape_y}")

    return X, y



# Function to generate a frame with a given velocity
def generate_frame(base_frame, velocity):
    return base_frame + velocity


def generate_simple_frames():
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

    # Visualize one data sample of X and y
    sample_index = 0

    plt.figure(figsize=(15, 6))

    # Plot X channels
    for i in range(X.shape[1]):
        plt.plot(X[sample_index, i, :], label=f'X Channel {i + 1}')

    # Plot y channels
    for i in range(y.shape[1]):
        plt.plot(y[sample_index, i, :], label=f'y Channel {i + 1 + X.shape[1]}')

    plt.title('Visualization of One Data Sample (X and y)')
    plt.xlabel('Frame Length')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def generate_base_frame(frame_length, car_width):
    base_frame = np.zeros(frame_length)
    
    # Calculate the starting position of the car
    # car_start = (frame_length - car_width) // 2
    car_start = frame_length - car_width-5

    
    # Create a rectangle representing the car
    base_frame[car_start:car_start + car_width] = 10
    
    return base_frame

# Function to generate a frame with movement
def generate_moved_frame(base_frame, horizontal_velocity, vertical_velocity):
    # Copy the base frame
    moved_frame = np.copy(base_frame)
    
    # Move the frame based on the velocities
    if horizontal_velocity > 0:
        # Check if moving would exceed the frame size
        if np.any(moved_frame[:-horizontal_velocity]):
            # If so, move to the rightmost position
            moved_frame = np.roll(base_frame, shift=-horizontal_velocity)
        else:
            # Otherwise, move as usual
            moved_frame[horizontal_velocity:] = base_frame[:-horizontal_velocity]
    elif horizontal_velocity < 0:
        # Check if moving would exceed the frame size
        if np.any(moved_frame[-horizontal_velocity:]):
            # If so, move to the leftmost position
            moved_frame = np.roll(base_frame, shift=-horizontal_velocity)
        else:
            # Otherwise, move as usual
            moved_frame[:horizontal_velocity] = base_frame[-horizontal_velocity:]
    
    # Increase all values when moving vertically
    moved_frame += vertical_velocity
    
    return moved_frame


# Function to visualize frames
def visualize_frames(frames, title='Sequence of Frames with Car Movement', save_path=None):
    plt.figure(figsize=(60, 25))
    for i, frame in enumerate(frames):
        plt.plot(frame, label=f'Frame {i + 1}')

        # Set y-axis limits and aspect ratio
        plt.ylim([-1, np.max(frames) + 1])
        plt.gca().set_aspect('equal', adjustable='box')

        plt.title(title)
        plt.xlabel('Position')
        plt.ylabel('Value')
        plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

