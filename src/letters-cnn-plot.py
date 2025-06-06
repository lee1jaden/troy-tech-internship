import numpy as np
import pandas as pd
from keras import models
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from letters import index_to_letter


def main():
    # Import and sample the data.
    print("Importing data...")
    data = pd.read_csv("./data/A_Z Handwritten Data.csv").astype("int")
    images = data.sample(n=12)
    labels = images.pop(images.columns[0])
    images = np.reshape(images, (images.shape[0], 28, 28, 1))

    # Predict the letters using the model.
    print("Loading model...")
    model: models.Sequential = models.load_model("./models/letters-cnn.keras")
    probabilities = model.predict(images)

    # Display test images with their predictions.
    print("Producing plots...")
    fig, axes = plt.subplots(3, 4)
    for i, ax in enumerate(axes.flatten()):
        chart: Axes = ax
        img = images[i].reshape((28, 28))
        chart.imshow(img, cmap="Greys")
        prediction = index_to_letter(np.argmax(probabilities[i]))
        actual = index_to_letter(labels.iloc[i])
        chart.set_title(f"Prediction: {prediction}\nActual: {actual}")
        chart.set_axis_off()
    plt.savefig("./examples/letters-cnn.png")


if __name__ == "__main__":
    main()
