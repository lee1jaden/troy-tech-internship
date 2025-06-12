import cv2
import imutils
from imutils.contours import sort_contours
import numpy as np
from keras import models
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from characters import index_to_char


def format_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (20, 20), interpolation=cv2.INTER_LINEAR)
    image[image > 127] = 255
    image = cv2.copyMakeBorder(
        image,
        4,
        4,
        4,
        4,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    return np.reshape(image, (28, 28, 1))


def main():
    print("Processing image...")
    image = cv2.imread("./data/namebirthday.jpg")
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.GaussianBlur(processed_image, (5, 5), cv2.BORDER_CONSTANT)
    processed_image = cv2.threshold(processed_image, 105, 255, 0)[1]
    processed_image = processed_image.astype("uint8")
    processed_image = cv2.Canny(processed_image, 40, 85)
    # cv2.imwrite("./examples/words-processed.png", processed_image)

    print("Analyzing contours...")
    contour_image = image.copy()
    contours = imutils.grab_contours(
        cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    )
    contours = sort_contours(contours, method="left-to-right")[0]
    # output = cv2.drawContours(image, contours, -1, (0, 0, 255), 5)
    letter_images = []
    for cntr in contours:
        (x, y, w, h) = cv2.boundingRect(cntr)
        if (w >= 50 and w <= 500) and (h >= 110 and h <= 1000):
            cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 0, 255), 5)
            letter = format_image(image[y : y + h, x : x + w])
            letter_images.append(letter)

    print("Predicting letters...")
    model: models.Sequential = models.load_model("./models/characters.keras")
    probabilities = model.predict(np.array(letter_images))
    fig, axes = plt.subplots(4, 4, figsize=(6.4, 6.4))
    axes = axes.flatten()
    for i in range(len(letter_images)):
        chart: Axes = axes[i]
        chart.imshow(letter_images[i], cmap="Greys_r")
        prediction = index_to_char(np.argmax(probabilities[i]))
        chart.set_title(f"Prediction: {prediction}")
        chart.set_axis_off()
    fig.savefig("./examples/words.png")
    cv2.imwrite("./examples/words-contours.png", contour_image)


if __name__ == "__main__":
    main()
