import numpy as np
import matplotlib.pyplot as plt

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
