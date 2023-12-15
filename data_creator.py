import os
import numpy as np
import matplotlib.pyplot as plt


def get_X_y(prev_frames, future_frames):
    X_arr=[]
    y_arr=[]
    directory_path = '../YOLOPv2-1D_Coordinates/data_npy'

    filenames = os.listdir(directory_path)
    print(filenames)

    for _file in filenames:
        file = directory_path+"/"+_file
        X_file = np.load(file)

        window_size = prev_frames + future_frames
        X = [X_file[i:i+prev_frames] for i in range(len(X_file[:-window_size]))]
        y = [[X_file[i+prev_frames+future_frames-1]] for i in range(len(X_file[:-window_size]))]

        X_arr.extend(X)
        y_arr.extend(y)
    
    X_arr = np.array(X_arr)
    y_arr = np.array(y_arr)

    shape_X = X_arr.shape
    shape_y = y_arr.shape
    print(f"Shape of X: {shape_X}")
    print(f"Shape of y: {shape_y}")

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