from src.models.conv_lstm import ConvLSTM1D
from src.pipeline import train
from src.dataset import SimpleFrameDataset, VideoFrameDataset
import torch.nn as nn
import torch
from src.config import Config

config = Config()

input_size = 100
hidden_size = 500
kernel_size = 3
num_layers = 3
learning_rate = 0.001
model = ConvLSTM1D(input_size, hidden_size, kernel_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if config.dataset_type == 'simple':
    dataset = SimpleFrameDataset(
        frame_length=100,
        car_width=20,
        min_velocity=1,
        max_velocity=5,
        num_frames=100,
        window_x=10,
        window_y=5,
        num_duplicates=20,
        train_frame_start=0,
        val_frame_start=60,
        test_frame_start=80,
        horizontal_velocity=1,
        vertical_velocity=1,
        acceleration=False,
    )
elif config.dataset_type == 'video':
    dataset = VideoFrameDataset(
        directory_path=config.dataset_path,
        split_ratio=0.80,
        test_flag=config.test_flag,
        DRR=config.DRR,
        n_th_frame=config.n_th_frame,
        frame_avg_rate=config.frame_avg_rate,
        prev_frames=config.prev_f,
        future_frames=config.future_f,
    )

print(config.model_name)
train(
    dataset,
    criterion,
    optimizer,
    model,
    config.model_name,
    config.train_flag,
    config.test_flag,
    config.n_th_frame,
    config.future_f,
    num_epochs=1000,
    batch_size=25
)