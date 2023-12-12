import os
import numpy as np


def get_X_y(prev_frames, future_frames):
    X_files=[]
    y_files=[]
    directory_path = '../YOLOPv2-1D_Coordinates/train_data/sequential'

    filenames = os.listdir(directory_path)
    print(filenames)

    for _file in filenames:
        file = directory_path+"/"+_file
        X_file = np.load(file)
        X_files.extend(X_file)

    window_size = prev_frames * future_frames
    # X = [X_files[i:i+prev_frames] for i in range(len(X_files[:-window_size]))]
    X,y = [], []
    for i in range(len(X_files[:-window_size-future_frames])):
        # if i % future_frames == 0:
        temp=[]
        for x in range(i,i+window_size,future_frames):
            temp.append(X_files[x])
            # X.append(X_files[x])
        # print(len(temp))
        X.append(temp)
        y.append(X_files[i+window_size+future_frames])
    print(len(temp))

    # y = [[X_files[i+prev_frames+future_frames-1]] for i in range(len(X_files[:-window_size]))]
    
    print(len(X_files))
    print(len(X))
    print(len(y))

    return X, y

# get_X_y(10, 5)