import os
import tensorflow as tf # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from sklearn.model_selection import train_test_split # type: ignore



class DataLoader():
    def __init__(self, train_csv, test_csv, train_dir, test_dir, img_size = (128,128), batch_size = 32 , *args):
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def load_data(self):
        # load the csv
        train_df = pd.read_csv(self.train_csv)
        test_df = pd.read_csv(self.test_csv)

        train_df["image_path"] = train_df['file_name'].apply(lambda x: os.path.join(self.train_dir, x))
        test_df["image_path"] = test_df['id'].apply(lambda x: os.path.join(self.test_dir, x))

        # split it
        X = train_df['image_path'].values
        y = train_df['label'].values

        x_train, x_val , y_train, y_val = train_test_split(X,y, test_size=0.2,random_state=42)

        self.train_df = pd.DataFrame({'image_path': x_train,'label': y_train})
        self.val_df = pd.DataFrame({'image_path': x_val,'label': y_val})
        self.test_df = test_df

    def _process_image(self, image_path):
        # load and process the image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image,self.img_size)
        # normalize the image
        image = image/255.0
        return image

    def __len__(self):
        pass
    def __getitem__(self,index):
        pass