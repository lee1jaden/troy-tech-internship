import numpy as np
import pandas as pd
from keras import models
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# Import and sample the data.
print("Importing data...")
data = pd.read_csv("./data/A_Z Handwritten Data.csv").astype("int")
images = data.sample(n=12)
labels = images.pop(images.columns[0])

# Predict the letters using the model.
print("Loading model...")
model: models.Sequential = models.load_model("./models/characters.keras")
probabilities = model.predict(images)


def index_to_letter(index: int) -> str:
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[index]


# Display test images with their predictions.
print("Producing plots...")
fig, axes = plt.subplots(3, 4)
for i, ax in enumerate(axes.flatten()):
    chart: Axes = ax
    img = images.iloc[i].to_numpy().reshape((28, 28))
    chart.imshow(img, cmap="Greys")
    prediction = index_to_letter(np.argmax(probabilities[i]))
    actual = index_to_letter(labels.iloc[i])
    chart.set_title(f"Prediction: {prediction}\nActual: {actual}")
    chart.set_axis_off()
plt.savefig("./examples/characters.png")
