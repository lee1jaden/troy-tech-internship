import numpy as np
import pandas as pd
from keras import models
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from characters import index_to_char, retrieve_char_data


def main():
    # Import and sample the data.
    print("Importing data...")
    images, labels = retrieve_char_data()
    sample = np.random.choice(np.arange(0, labels.shape[0]), size=12, replace=False)
    images = np.reshape(images[sample], (12, 28, 28, 1))
    labels = labels[sample]

    # Predict the letters using the model.
    print("Loading model...")
    model: models.Sequential = models.load_model("./models/characters.keras")
    probabilities = model.predict(images)

    # Display test images with their predictions.
    print("Producing plots...")
    fig, axes = plt.subplots(3, 4)
    for i, ax in enumerate(axes.flatten()):
        chart: Axes = ax
        chart.imshow(images[i], cmap="Greys")
        prediction = index_to_char(np.argmax(probabilities[i]))
        actual = index_to_char(labels[i])
        chart.set_title(f"Prediction: {prediction}\nActual: {actual}")
        chart.set_axis_off()
    plt.savefig("./examples/characters.png")


if __name__ == "__main__":
    main()
