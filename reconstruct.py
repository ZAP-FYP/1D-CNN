import numpy as np  

last_test_x_frames = np.load('last_frame_of_x.npy')
merged_predicted_tes_y_numpy = np.load('merged_predicted_tes_y.npy')
y1 = np.load('y1.npy')
X = np.load('X.npy')

count, in_channels, in_seq_len = X.shape
idx = int(count * 0.80)

last_test_x_frames = last_frame_of_x[idx:]

print("final_x_last ",last_test_x_frames.shape)
print("final_y ",merged_predicted_tes_y_numpy.shape)

reconstructed_frames = np.zeros_like(merged_predicted_tes_y_numpy)

# Loop through each data point
for data_point_index in range(merged_predicted_tes_y_numpy.shape[0]):
    cumulative_frame = last_test_x_frames[data_point_index]
    # Loop through each frame difference in the data point
    for frame_diff_index in range(merged_predicted_tes_y_numpy.shape[1]):
        frame_diff = merged_predicted_tes_y_numpy[data_point_index, frame_diff_index]
        cumulative_frame = cumulative_frame + frame_diff
        reconstructed_frames[data_point_index, frame_diff_index] = cumulative_frame

print("reconstructed_frames ",reconstructed_frames.shape)
np.save(f'reconstructed_frames.npy', reconstructed_frames)
y1_test = y1[idx:]
np.save(f'y1_test.npy', y1_test)
print("y1_test ",y1_test.shape)