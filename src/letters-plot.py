import numpy as np
import pandas as pd
import cv2
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

    # Predict the letters using the model.
    print("Loading model...")
    model: models.Sequential = models.load_model("./models/letters.keras")
    probabilities = model.predict(images)

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
    plt.savefig("./examples/letters.png")


def frequency_distrubition():
    print("Reading data...")
    data = pd.read_csv("./data/A_Z Handwritten Data.csv").astype("int")
    labels = data["0"]
    letters = [index_to_letter(i) for i in range(26)]
    counts = labels.value_counts(sort=False).values

    print("Making chart...")
    plt.figure()
    plt.bar(letters, counts, color="skyblue")
    plt.xlabel("Letter")
    plt.ylabel("Count")
    plt.title("Frequencies of Letters in Dataset")
    plt.grid()
    plt.savefig("./examples/letters-distribution.png")


def predict_image():
    print("Processing image")
    image = cv2.imread("./data/handwritten_letter.jpg")

    processed_image = image.copy()
    processed_image = cv2.GaussianBlur(processed_image, (5, 5), 0)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    # _, processed_image = cv2.threshold(processed_image, 100, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("./examples/processed_image.png", processed_image)

    img_final = cv2.resize(img_thresh, (28, 28))
    # img_final = np.reshape(img_final, (1, 28, 28, 1))

    # model: models.Sequential = models.load_model("./models/letters.keras")
    # prediction = index_to_letter(np.argmax(model.predict(img_final)))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (400, 440))

    # cv2.putText(
    #     image,
    #     "Dataflair _ _ _ ",
    #     (20, 25),
    #     cv2.FONT_HERSHEY_TRIPLEX,
    #     0.7,
    #     color=(0, 0, 230),
    # )
    # cv2.putText(
    #     image,
    #     "Prediction: " + img_pred,
    #     (20, 410),
    #     cv2.FONT_HERSHEY_DUPLEX,
    #     1.3,
    #     color=(255, 0, 30),
    # )
    # cv2.imshow("Dataflair handwritten character recognition _ _ _ ", image)

    # while 1:
    #     k = cv2.waitKey(1) & 0xFF
    #     if k == 27:
    #         break
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    predict_image()
