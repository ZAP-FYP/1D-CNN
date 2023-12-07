import numpy as np
import cv2
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from decouple import config
from datetime import datetime
from torchsummary import summary
import sys

model_name = "Simple-data-CNN"
checkpoint_file = 'model/'+model_name+'/model_checkpoint.pth'
if not os.path.exists("model/"+model_name):
    os.makedirs("model/"+model_name)

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

f = open('model/'+model_name+'/log.txt', 'w')
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)



class VideoDataset(Dataset):
    def __init__(self, X, y):
        self.x = torch.from_numpy(X).float().requires_grad_()
        self.y = torch.from_numpy(y).float().requires_grad_()
        self.n_samples = y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples



class ConvNet(nn.Module):
    
    def __init__(self, in_channels, in_seq_len):
        super(ConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=2),
            nn.ReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(5*100, 500)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        
        return x
 


npy_file = np.load('../YOLOPv2-1D_Coordinates/data_npy/frame_coords_IMG_0260.npy')

X00 = npy_file[0]
X10 = npy_file[1]
X20 = npy_file[3]

y0 = npy_file[2]
y1 = npy_file[4]
y2 = npy_file[6]

X0, X1, X2 = [], [], []
for i in range(10):
    X0.append(X00)
    
for i in range(10):
    X1.append(X10)

for i in range(10):
    X2.append(X20)

X, y = [], []
for i in range(35):
    X.append(X0)
    y.append(y0)
    X.append(X1)
    y.append(y1)
    X.append(X2)
    y.append(y2)

np.random.shuffle(X)
np.random.shuffle(y)

print(len(X))
print(len(X[0]))
print(len(X[0][0]))

print(len(y))
print(len(y[0]))

for i in range(len(X)):
    plt.plot(X[i][0], label=f"Label Array_{i}")

# plt.legend()
plt.title(f"Frame")
plt.savefig(os.path.join("sample_viz.png"))
plt.close()

