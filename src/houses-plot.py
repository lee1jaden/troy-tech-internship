import numpy as np
import pandas as pd
from keras import models
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from houses import import_data, process_image, index_to_number


def main():
    # Import and sample the data.
    images, labels = import_data()
    sample = np.random.choice(np.arange(0, labels.shape[0]), size=12, replace=False)
    images = images[sample]
    processed_images = []
    for img in images:
        processed_images.append(process_image(img))
    processed_images = np.reshape(processed_images, (12, 32, 32, 1))
    labels = labels[sample]

    # Predict the house numbers using the model.
    print("Loading model...")
    model: models.Sequential = models.load_model("./models/houses.keras")
    probabilities = model.predict(processed_images)

    # Display test images with their predictions.
    print("Producing plots...")
    fig, axes = plt.subplots(3, 4)
    for i, ax in enumerate(axes.flatten()):
        chart: Axes = ax
        img = images[i]
        chart.imshow(img, cmap="Greys")
        prediction = index_to_number(np.argmax(probabilities[i]))
        actual = index_to_number(labels[i].item())
        chart.set_title(f"Prediction: {prediction}\nActual: {actual}")
        chart.set_axis_off()
    plt.savefig("./examples/houses.png")


def frequency_distrubition():
    images, labels = import_data()
    labels = pd.Series(np.reshape(labels, (labels.shape[0])))
    counts = labels.value_counts(sort=False).values
    numbers = [index_to_number(i) for i in range(1, 11)]

    print("Making chart...")
    plt.figure()
    plt.bar(numbers, counts, color="skyblue")
    plt.xlabel("Number")
    plt.ylabel("Count")
    plt.title("Frequencies of Numbers in Dataset")
    plt.grid()
    plt.savefig("./examples/houses-distribution.png")


if __name__ == "__main__":
    main()
