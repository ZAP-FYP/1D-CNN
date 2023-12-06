import os
import numpy as np


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

get_X_y(10, 5)