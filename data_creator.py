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
    
    print("X_len:", len(X_arr))
    print("y_len:", len(y_arr))

    # print("X")
    # print(X_arr[0])
    # print("y")
    # print(y_arr[0])

    return X, y

# get_X_y(10, 5)

def get_simplest_dataset():
    npy_file = np.load('../YOLOPv2-1D_Coordinates/data_npy/frame_coords_IMG_0260.npy')

    X00 = npy_file[0]

    X, y = [], []

    for i in range(1000):
        X.append([X00])
        y.append([X00])
    
    plt.plot(X00, label=f"Label Array_1")

    # plt.legend()
    plt.title(f"Frame")
    plt.savefig(os.path.join("sample_viz_1.png"))
    plt.close()

    return X, y

def get_simple_dataset():
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

    # np.random.shuffle(X)
    # np.random.shuffle(y)

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

    return X, y


get_simplest_dataset()