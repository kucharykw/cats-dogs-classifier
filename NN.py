import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
import warnings

warnings.filterwarnings("ignore")

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3  # depth

train_dir = "data/train"
filenames = os.listdir(train_dir)
targets = []

for filename in filenames:
    target = filename.split(".")[0]
    if target == "cat":
        targets.append(0)  # 0 means a cat
    else:
        targets.append(1)  # 1 means a dog

dataset = pd.DataFrame({
          "filename": filenames,
          "target": targets
          })

dataset['target'].value_counts().plot(kind='bar');

sample_image_file = random.choice(filenames)
sample_image = cv2.imread(os.path.join(train_dir, sample_image_file))
cv2.imshow("Sample image", sample_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

X_train = []
y_train = []


#BUILD MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout, Flatten,
                                     Dense, Activation, BatchNormalization)


# DATA PREPROCESSING
from tensorflow.keras.preprocessing import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
# df["category"] = df["category"].replace({0: "cat", 1: "dog"}) - nice trick
train_data, validate_data = train_test_split()