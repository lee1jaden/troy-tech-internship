import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import models, layers, utils

# Import and organize the data.
print("Importing data...")
data = pd.read_csv("./data/A_Z Handwritten Data.csv").astype("int")
labels = data["0"]
images = data.drop("0", axis=1)
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2
)

# Build and fit the neural network model.
print("Building model...")
model = models.Sequential()
model.add(layers.Dense(784, activation="relu"))
model.add(layers.Dense(784, activation="relu"))
model.add(layers.Dense(26, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, utils.to_categorical(train_labels), epochs=5, batch_size=128)
model.save("./models/characters.keras")

# Display metrics for the model.
print("Evaluating network...")
loss, accuracy = model.evaluate(test_images, utils.to_categorical(test_labels))
