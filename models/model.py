# import libaries
import os
import pandas as pd # type: ignore
from data_loader import DataLoader
# define constant
BASE = ''

# subdirectory or file to the base
models_path = os.path.join(BASE, 'models')
datasets_path = os.path.join(BASE, 'dataset')
print("Models Path:", models_path)
print("Datasets Path:", datasets_path)
# Get the file name of images
train_file_name = f"{datasets_path}/train.csv"
test_file_name = f"{datasets_path}/test.csv"



if __name__=="__main__":
    dataloader =  DataLoader(train_csv=train_file_name, test_csv=test_file_name, train_dir = datasets_path, test_dir = datasets_path)
    dataloader.load_data()

    path  = dataloader.train_df.image_path[0]

    print(dataloader._process_image(path))
    

