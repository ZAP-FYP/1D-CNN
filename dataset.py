import os
import numpy as np
import matplotlib.pyplot as plt

# Load the data
X = np.load('finalX1.npy')
y = np.load('finaly1.npy')

# Create a folder for saving visualizations
save_folder = 'dataset_visual'
os.makedirs(save_folder, exist_ok=True)

# Visualize the first 5 samples
num_samples_to_visualize = 10

for i in range(num_samples_to_visualize):
    # Plot the frames in X and y side by side
    plt.figure(figsize=(15, 5))
    
    # Plot frames in X
    for j in range(9):
        plt.subplot(2, 9, j + 1)
        plt.plot(X[i, j, :])  # Assuming your frames are 1D arrays
        plt.title(f'Input {j + 1}')
        plt.xlabel('Time')
        plt.ylabel('Feature')
    
    # Plot frames in y
    for j in range(5):
        plt.subplot(2, 9, 9 + j + 1)
        plt.plot(y[i, j, :])  # Assuming your frames are 1D arrays
        plt.title(f'Output {j + 1}')
        plt.xlabel('Time')
        plt.ylabel('Feature')
    
    plt.suptitle(f'Sample {i + 1}: Input Frames (X) and Output Frames (y)')
    plt.tight_layout()

    # Save the visualization
    save_path = os.path.join(save_folder, f'sample_{i + 1}_visualization.png')
    plt.savefig(save_path)
    plt.close()  # Close the figure to release memory

print(f"Visualizations saved in the '{save_folder}' folder.")
