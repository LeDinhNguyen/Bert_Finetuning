import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical

# Instantiating a small convnet
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))

# Adding a classifier on the top of the convnet
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

# Training the convnet on the MNIST images
## load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_labels = train_labels.astype("float32") / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_labels = test_labels.astype("float32") / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(
    optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

history = model.fit(train_images, train_labels, epochs=10, batch_size=64)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test acc: ", test_acc)

# Model Summary
# model.summary()
