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
    sample = np.random.choice(
        np.arange(0, test_labels.shape[0]), size=12, replace=False
    )
    images = test_images[sample]
    labels = test_labels[sample]

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


def frequency_distrubition():
    print("Reading data...")
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    labels = pd.Series(np.concatenate([train_labels, test_labels]))
    letters = [index_to_digit(i) for i in range(10)]
    counts = labels.value_counts(sort=False).values

    print("Making chart...")
    plt.figure()
    plt.bar(letters, counts, color="skyblue")
    plt.xlabel("Digit")
    plt.ylabel("Count")
    plt.title("Frequencies of Digit in Dataset")
    plt.grid()
    plt.savefig("./examples/digits-distribution.png")


if __name__ == "__main__":
    main()
