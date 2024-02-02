# Prediction Models for Drivable Area

## Description

This repository contains the deep learning models and techniques used to predict the changes in the drivable area of an ego vehicle. The input is an array of 100 y-coordinates representing the boundary of the drivable area polygon. The output is an array representing the predicted drivable area for the recent future. 

## How to Run

1. First, install the dependencies.
   ```
   pip install -r requirements.txt
   ```

2. Create a .env file using the env_template.txt file.

   You can set ``` N_TH_FRAME = True ``` to predict only for the nth frame in the future and not for all n frames in the future.
   Set ```DATASET_TYPE = "simple" ``` to generate simple drivable areas with consistent velocity, and set  ```DATASET_TYPE = "video" ``` to run for other drivable area datasets.
   Numpy files corresponding to other datasets can be found [here](https://drive.google.com/drive/u/1/folders/19Mszdhn1ZFpFtO027f2wmFwCrStYSoq5)

3. To run for other datasets, download the dataset from the above link and copy the local folder path to the ```directory_path``` variable under the ```VideoFrameDataset``` initiation in the run.py file.
   
4. Run the module
   ```
   python3 run.py
   ```

