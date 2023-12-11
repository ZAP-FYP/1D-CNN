import numpy as np
import matplotlib.pyplot as plt

# Sample data (replace with your actual data)
original_frames = np.load('y1_test.npy')  # 28 data points, each with 5 frames
reconstructed_frames_from_diff = np.load('reconstructed_frames.npy')  # 28 data points, each with 5 reconstructed frames
predicted_frames = np.load('merged_predicted_y_numpy.npy')  # 28 data points, each with 5 additional reconstructed frames

# Loop through each data point
for data_point_index in range(28):
    data_point_original = original_frames[data_point_index]
    data_point_reconstructed = reconstructed_frames_from_diff[data_point_index]
    data_point_reconstructed_1 = predicted_frames[data_point_index]

    # Create a figure with 5 subplots
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))
    
    for i in range(5):
        axes[i].plot(data_point_original[i], label=f'Original Frame {i+1}')
        axes[i].plot(data_point_reconstructed[i], label=f'Reconstructed Frame {i+1}')
        axes[i].plot(data_point_reconstructed_1[i], label=f'Predicted Frame {i+1}')
        axes[i].set_xlabel('Frame Index')
        axes[i].set_ylabel('Value')
        axes[i].legend()

    fig.suptitle(f'Data Point {data_point_index + 1}')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save the figure as an image (you can specify the file format)
    plt.savefig(f'data_point_{data_point_index + 1}.png')

# Show or save each graph as needed
plt.show()
