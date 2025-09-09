import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
import cv2
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from keras.layers import Dense, Conv2D, BatchNormalization, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Reshape
from keras.layers import Flatten
# Supress info, warnings and error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import os
from sklearn.model_selection import train_test_split
import shutil

def split_data(original_folder, train_folder, test_folder, split_ratio=0.8, random_seed=42):
    # Create train and test folders if they don't exist
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Get the list of all files in the original folder
    all_files = os.listdir(original_folder)
    
    # Split the data into training and testing sets
    train_files, test_files = train_test_split(all_files, test_size=1 - split_ratio, random_state=random_seed)

    # Copy files to the train folder
    for file in train_files:
        src_path = os.path.join(original_folder, file)
        dest_path = os.path.join(train_folder, file)
        shutil.copy(src_path, dest_path)

    # Copy files to the test folder
    for file in test_files:
        src_path = os.path.join(original_folder, file)
        dest_path = os.path.join(test_folder, file)
        shutil.copy(src_path, dest_path)

# Example usage:
original_folder = "/home/213112002/Atelectasis/Dataset/Normal/"
train_folder = "/home/213112002/Atelectasis/Dataset/training/Normal"
test_folder = "/home/213112002/Atelectasis/Dataset/testing/Normal"

split_data(original_folder, train_folder, test_folder, split_ratio=0.8, random_seed=42)