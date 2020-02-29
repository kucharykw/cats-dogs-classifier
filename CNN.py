import warnings
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout, Flatten,
                                     Dense, Activation, BatchNormalization)
warnings.filterwarnings("ignore")

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 1  # read in grayscale

train_dir = "data/train"
test_dir = "data/test1"
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

dataset['target'].value_counts().plot(kind='bar')

#
# sample_image_file = random.choice(filenames)
# sample_image = cv2.imread(os.path.join(train_dir, sample_image_file))
# cv2.imshow("Sample image", sample_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

X_train = []
y_train = []

for filename in filenames:
    X_train.append(cv2.resize(cv2.imread(os.path.join(train_dir,
                   filename), cv2.IMREAD_GRAYSCALE), IMAGE_SIZE,
                   interpolation=cv2.INTER_CUBIC))

# Train on smaller dataset
dataset_size = 6000
X_train = X_train[:dataset_size]
X_train = np.array(X_train).reshape(-1, 128, 128, 1)
X_train = X_train / 255
y_train = dataset["target"].values
y_train = y_train[:dataset_size]
y_train = to_categorical(y_train)


model = Sequential()
model.add(Conv2D(64, (3, 3), activation="relu",
                 input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="RMSprop",
              metrics=["accuracy"])
model.summary()
history = model.fit(X_train, y_train, epochs=5, batch_size=32,
                    validation_split=0.2)
