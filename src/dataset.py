import torch
from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt


class DrivableAreaDataset(Dataset):
    def __init__(self, X, y):
        self.x = torch.from_numpy(X).float().requires_grad_()
        self.y = torch.from_numpy(y).float().requires_grad_()
        self.n_samples = y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class SimpleFrameDataset:
    def __init__(
        self,
        frame_length,
        car_width,
        min_velocity,
        max_velocity,
        num_frames,
        window_x,
        window_y,
        num_duplicates=20,
        train_frame_start=None,
        val_frame_start=None,
        test_frame_start=None,
        val_test_frames=20,
        horizontal_velocity=1,
        vertical_velocity=1,
        acceleration=False,
    ):
        self.frame_length = frame_length
        self.car_width = car_width
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.num_frames = num_frames
        self.window_x = window_x
        self.window_y = window_y
        self.num_duplicates = num_duplicates
        self.train_frame_start = (
            train_frame_start if train_frame_start is not None else 0
        )
        self.val_frame_start = val_frame_start
        self.test_frame_start = test_frame_start
        self.horizontal_velocity = horizontal_velocity
        self.vertical_velocity = vertical_velocity
        self.acceleration = acceleration

        self.create_dataset()

    @staticmethod
    def generate_base_frame(frame_length, car_width):
        base_frame = np.zeros(frame_length)
        car_start = frame_length - car_width - 5
        base_frame[car_start : car_start + car_width] = 10
        return base_frame

    @staticmethod
    def generate_moved_frame(
        base_frame,
        horizontal_velocity,
        vertical_velocity,
        noise_factor=0.1,
        noise_flag=False,
    ):
        moved_frame = np.copy(base_frame)

        if horizontal_velocity > 0:
            if np.any(moved_frame[:-horizontal_velocity]):
                moved_frame = np.roll(base_frame, shift=-horizontal_velocity)
            else:
                moved_frame[horizontal_velocity:] = base_frame[:-horizontal_velocity]
        elif horizontal_velocity < 0:
            if np.any(moved_frame[-horizontal_velocity:]):
                moved_frame = np.roll(base_frame, shift=-horizontal_velocity)
            else:
                moved_frame[:horizontal_velocity] = base_frame[-horizontal_velocity:]

        moved_frame += vertical_velocity
        if noise_flag:
            noise = np.random.normal(scale=noise_factor, size=moved_frame.shape)
            moved_frame += noise

        return moved_frame

    def generate_frames(self):
        base_frame = self.generate_base_frame(self.frame_length, self.car_width)
        frames = [base_frame]

        if self.acceleration:
            acceleration_frames = 300
            constant_velocity_frames = 0
            deceleration_frames = 0

            pattern = np.concatenate(
                [
                    np.linspace(
                        self.min_velocity, self.max_velocity, acceleration_frames
                    ),
                    np.full(constant_velocity_frames, self.max_velocity),
                    np.linspace(
                        self.max_velocity, self.min_velocity, deceleration_frames
                    ),
                ]
            )

            velocity_sequence = np.tile(pattern, self.num_frames // len(pattern))
        else:
            velocity_sequence = np.full(self.num_frames, self.vertical_velocity)

        for i in range(self.num_frames):
            frame = self.generate_moved_frame(
                frames[-1],
                self.horizontal_velocity,
                velocity_sequence[i],
                noise_flag=True,
            )
            frames.append(frame)

        self.frames = frames

    def split_frames(self):
        train_frames, val_frames, test_frames = self.get_split_frames()

        self.x_train, self.y_train = self.index_frames(train_frames)
        self.x_val, self.y_val = self.index_frames(train_frames[30:50])
        self.x_test, self.y_test = self.index_frames(train_frames[40:60])

    def get_split_frames(self):
        train_frames = self.frames[self.train_frame_start : self.val_frame_start]
        val_frames = self.frames[self.val_frame_start : self.test_frame_start]
        test_frames = self.frames[self.test_frame_start :]

        return train_frames, val_frames, test_frames

    def index_frames(self, frames):
        x_frames = np.array(
            [
                frames[i : i + self.window_x]
                for i in range(len(frames) - self.window_x - self.window_y)
            ]
        )
        y_frames = np.array(
            [
                frames[i + self.window_x : i + self.window_x + self.window_y]
                for i in range(len(frames) - self.window_x - self.window_y)
            ]
        )
        return x_frames, y_frames

    def create_numpy_arrays(self):
        self.x_train_duplicate = np.tile(self.x_train, (self.num_duplicates, 1, 1))
        self.y_train_duplicate = np.tile(self.y_train, (self.num_duplicates, 1, 1))

        self.flatten_y = self.y_train_duplicate.reshape(
            (len(self.y_train_duplicate), -1)
        )
        self.flatten_y_val = self.y_val.reshape((len(self.y_val), -1))
        self.flatten_y_test = self.y_test.reshape((len(self.y_test), -1))

        count, in_channels, in_seq_len = self.x_train_duplicate.shape

    def visualize_frames(
        self, save_directory="frames_plots", save_filename="frames_plot.png"
    ):
        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(save_directory, save_filename)
        plt.figure(figsize=(60, 25))
        for i, frame in enumerate(self.frames):
            plt.plot(frame, label=f"Frame {i + 1}")
            plt.ylim([-1, np.max(self.frames) + 1])
            plt.gca().set_aspect("equal", adjustable="box")
            plt.title("Sequence of Frames with Car Movement")
            plt.xlabel("Position")
            plt.ylabel("Value")
            plt.legend()

        if save_path:
            plt.savefig(save_path)
            print(f"Figure saved to: {save_path}")
        else:
            plt.show()

    def create_dataset(self):
        self.generate_frames()
        self.split_frames()
        self.create_numpy_arrays()
        self.train_dataset = DrivableAreaDataset(self.x_train_duplicate, self.flatten_y)
        self.validation_dataset = DrivableAreaDataset(self.x_val, self.flatten_y_val)
        self.test_dataset = DrivableAreaDataset(self.x_test, self.flatten_y_test)


class VideoFrameDataset:
    def __init__(
        self,
        directory_path,
        split_ratio=0.80,
        test_flag=False,
        DRR=0,
        n_th_frame=True,
        do_frame_averaging=False,
        prev_frames=10,
        future_frames=5,
    ):
        self.directory_path = directory_path
        self.split_ratio = split_ratio
        self.test_flag = test_flag
        self.DRR = DRR
        self.n_th_frame = n_th_frame
        self.do_frame_averaging = do_frame_averaging
        self.prev_frames = prev_frames
        self.future_frames = future_frames

        self.create_dataset()

    def get_X_y(self, data, prev_frames, future_frames):
        
        window_size = prev_frames + future_frames
        X = [data[i : i + prev_frames] for i in range(len(data[:-window_size]))]

        if self.n_th_frame:
            y = [
                data[i + prev_frames + future_frames]
                for i in range(len(data[:-window_size]))
            ]
        else:
            y = [
                data[(i + prev_frames) : (i + prev_frames + future_frames)]
                for i in range(len(data[:-window_size]))
            ]

        return np.array(X), np.array(y)

    def create_averaged_frames(self, data, drr):
        averaged_frames = []
        
        for i in range(0, len(data), drr):
            frames_to_average = [
                i + drr // 4,
                i + drr // 2,
                i + (3 * drr) // 4,
                i + drr - 1,
            ]
            valid_frames = [
                frame for frame in frames_to_average if 0 <= frame < len(data)
            ]
            non_zero_frames_data = [
                frame for frame in valid_frames if not np.all(data[frame] == 0)
            ]

            if non_zero_frames_data:
                averaged_frame = data[non_zero_frames_data].mean(axis=0)
                averaged_frames.append(averaged_frame)

        return np.array(averaged_frames)
    

    def create_dataset(self):
        data_npy = []

        filenames = [
            f for f in os.listdir(self.directory_path) if not f.startswith(".DS_Store")
        ]

        for _file in filenames:
            file = os.path.join(self.directory_path, _file)
            data_file = np.load(file)
            data_npy.extend(data_file)
        
        data_npy = np.array(data_npy)
        
        print("all frames numpy:", data_npy.shape)

        if self.do_frame_averaging:
            averaged_frames = self.create_averaged_frames(data_npy, self.DRR)
            X, y = self.get_X_y(averaged_frames, self.prev_frames, self.future_frames)
            print("averaged_frames:", averaged_frames.shape)
        else:
            X, y = self.get_X_y(data_npy, self.prev_frames, self.future_frames)
            print("Not averaging")
        
        print("averaged X and y:", X.shape, y.shape)


        flatten_y = y.reshape((len(y), -1))
        count, self.in_channels, self.in_seq_len = X.shape

        if not self.test_flag:
            idx = int(count)
        else:
            idx = int(count * self.split_ratio)

        val_idx = int(idx * self.split_ratio)

        if self.DRR == 0 or self.do_frame_averaging:
            self.train_dataset = DrivableAreaDataset(X[:val_idx:], flatten_y[:val_idx:])
            self.validation_dataset = DrivableAreaDataset(
                X[val_idx:idx:], flatten_y[val_idx:idx:]
            )
        else:
            self.train_dataset = DrivableAreaDataset(
                X[: val_idx : self.DRR], flatten_y[: val_idx : self.DRR]
            )
            self.validation_dataset = DrivableAreaDataset(
                X[val_idx : idx : self.DRR], flatten_y[val_idx : idx : self.DRR]
            )

        self.test_dataset = DrivableAreaDataset(
            X[idx :: self.DRR], flatten_y[idx :: self.DRR]
        )
        
        print(f'Train samples {len(self.train_dataset)}')
        print(f'Validation samples {len(self.validation_dataset)}')
        print(f'Test samples {len(self.test_dataset)}')

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
                plt.plot(
                    x_frame, label=f"Sample {sample_index}, Frame {frame_index} - Input (x)"
                )

            # Plot y
            for frame_index in range(num_frames_y):
                y_frame = y[sample_index, frame_index]
                plt.plot(
                    y_frame,
                    label=f"Sample {sample_index}, Frame {frame_index} - Output (y)",
                    color="red",
                )

            plt.xlabel("Time Steps")
            plt.ylabel("Values")
            plt.legend()

            plt.tight_layout()

            # Save the figure
            plt.savefig(
                os.path.join(sample_folder, f"sample_{sample_index}_visualization.png")
            )
            plt.close()
