import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from keras import models, layers, utils
import cv2 as cv2


def index_to_number(num: int) -> str:
    return str(num % 10)


def import_data():
    print("Downloading data...")
    data_1 = sio.loadmat("./data/train_32x32.mat")
    data_2 = sio.loadmat("./data/test_32x32.mat")
    images_1 = np.array(data_1["X"])
    images_2 = np.array(data_2["X"])
    images = np.concatenate((images_1, images_2), 3)
    images = np.moveaxis(images, -1, 0)
    labels_1 = np.array(data_1["y"])
    labels_2 = np.array(data_2["y"])
    labels = np.concatenate((labels_1, labels_2))
    return images, labels


def process_image(image):
    grayscale = np.dot(image, [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    return cv2.Canny(grayscale, 45, 80)


def main():
    images, labels = import_data()

    print("Processing images...")
    processed_images = np.empty((images.shape[0], 32, 32))
    for i in range(len(images)):
        processed_images[i] = process_image(images[i])
    processed_images = np.reshape(
        processed_images, (processed_images.shape[0], 32, 32, 1)
    )

    train_images, test_images, train_labels, test_labels = train_test_split(
        processed_images, labels, test_size=0.2
    )

    # Build the neural network model using convolutional, pooling, and typical neural network layers
    print("Training neural network...")
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")
    )
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(
        layers.Conv2D(
            filters=128, kernel_size=(3, 3), activation="relu", padding="valid"
        )
    )
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(11, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        train_images, utils.to_categorical(train_labels), epochs=5, batch_size=128
    )
    model.evaluate(test_images, utils.to_categorical(test_labels))
    model.save("./models/houses.keras")


if __name__ == "__main__":
    main()
