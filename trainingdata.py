import os
import numpy as np

# Define the folder paths for 'x' and 'y'
folder_x_path = './x'
folder_y_path = './y'

# Create empty lists to store the file names
file_names_x = []
file_names_y = []

# Iterate over the files in folder 'x'
for root, dirs, files in os.walk(folder_x_path):
    for file in files:
        if file.lower() != ".ds_store": 
            file_names_x.append(file)

# Iterate over the files in folder 'y'
for root, dirs, files in os.walk(folder_y_path):
    for file in files:
        if file.lower() != ".ds_store": 
            file_names_y.append(file)

# Sort the file names alphabetically
file_names_x_sorted = sorted(file_names_x)
file_names_y_sorted = sorted(file_names_y)

print(file_names_x_sorted)
print(file_names_y_sorted)

# Initialize variables to store the merged data for 'x' and 'y'
merged_data_x = None
merged_data_y = None

# Merge the contents of files in folder 'x'
for file_name in file_names_x_sorted:
    file_path = os.path.join(folder_x_path, file_name)
    try:
        data = np.load(file_path)
        if merged_data_x is None:
            merged_data_x = data
        else:
            merged_data_x = np.concatenate([merged_data_x, data], axis=0)
        print(f"Shape of data for file {file_name}: {data.shape}")
    except ValueError as e:
        print(f"Error loading or concatenating data for file: {file_name}")
        print(f"Error message: {e}")

# Print the shape of the merged_data_x array
print(f"Shape of merged_data_x after concatenation: {merged_data_x.shape}")



# Merge the contents of files in folder 'y'
for file_name in file_names_y_sorted:
    file_path = os.path.join(folder_y_path, file_name)
    data = np.load(file_path)
    if merged_data_y is None:
        merged_data_y = data
    else:
        merged_data_y = np.concatenate([merged_data_y, data], axis=0)

# Print the shape of the merged_data arrays for 'x' and 'y'
print("Shape of merged_data_x:", merged_data_x.shape)
print("Shape of merged_data_y:", merged_data_y.shape)

last_feature = merged_data_x[1:, -1, :]
print("Shape of last_feature:", last_feature.shape)

first_point = merged_data_x[0]
print("Shape of first_point:", first_point.shape)

prev_frames = np.concatenate((first_point,last_feature), axis=0)
print("Shape of merged_array:", prev_frames.shape)

# Initialize a list to store frame differences

prev_10_frames = []
next_5_frames = []

for i in range(len(prev_frames) - 15):  # Iterate through all frames except the last 15
    prev_10_frames.append(prev_frames[i:i+10])
    next_5_frames.append(prev_frames[i+10:i+15])
    
frame_diffs = []
prev_9_diff_frames = []
next_5_diff_frames = []

# Iterate through prev_frames to compute differences and visualize
for i in range(1, len(prev_frames)):
    frame_diff = [prev_frames[i][j] - prev_frames[i - 1][j] for j in range(len(prev_frames[i]))]
    frame_diffs.append(frame_diff)

for i in range(len(frame_diffs) - 14):  # Iterate through all frames except the last 14
    prev_9_diff_frames.append(frame_diffs[i:i+9])
    next_5_diff_frames.append(frame_diffs[i+9:i+14])


X = np.array(prev_10_frames)  # X.shape = (n, 10, 100)
print(X.shape)
y = np.array(next_5_frames)   # y.shape = (n, 5, 100)
print(y.shape)

last_feature_1 = X[:, -1, :]
# Save the extracted last feature as a new NumPy array in a new file
np.save(f'finallastframe.npy', last_feature_1)
print(last_feature_1.shape)

X1 = np.array(prev_9_diff_frames)  # X.shape = (n, 10, 100)
print(X1.shape)
y1 = np.array(next_5_diff_frames)   # y.shape = (n, 5, 100)
print(y1.shape)

np.save(f'finalX1.npy', X1)
np.save(f'finaly1.npy', y1)

np.save(f'finalX.npy', X)
np.save(f'finaly.npy', y)


