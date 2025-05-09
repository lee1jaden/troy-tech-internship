import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import models, layers, utils

# Import and organize the data.
print("Importing data...")
data = pd.read_csv("./data/A_Z Handwritten Data.csv").astype("float32")
labels = data["0"]
images = data.drop("0", axis=1)
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2
)

# Build and fit the neural network model.
print("Building model...")
network = models.Sequential()
network.add(layers.Dense(784, activation="relu"))
network.add(layers.Dense(784, activation="relu"))
network.add(layers.Dense(26, activation="softmax"))
network.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
network.fit(train_images, utils.to_categorical(train_labels), epochs=5, batch_size=128)
network.save_weights("./models/characters.weights.h5")

# Display metrics for the model.
print("Evaluating network...")
loss, accuracy = network.evaluate(test_images, utils.to_categorical(test_labels))
