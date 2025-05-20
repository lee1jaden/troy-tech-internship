import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras import models
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from digits import index_to_digit


def main():
    # Import and sample the data.
    print("Importing data...")
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    images = test_images[:12]
    labels = test_labels[:12]

    # Predict the letters using the model.
    print("Loading model...")
    model: models.Sequential = models.load_model("./models/digits.keras")
    probabilities = model.predict(images)

    # Display test images with their predictions.
    print("Producing plots...")
    fig, axes = plt.subplots(3, 4)
    for i, ax in enumerate(axes.flatten()):
        chart: Axes = ax
        chart.imshow(images[i], cmap="Greys")
        prediction = index_to_digit(np.argmax(probabilities[i]))
        actual = index_to_digit(labels[i])
        chart.set_title(f"Prediction: {prediction}\nActual: {actual}")
        chart.set_axis_off()
    plt.savefig("./examples/digits.png")


if __name__ == "__main__":
    main()
