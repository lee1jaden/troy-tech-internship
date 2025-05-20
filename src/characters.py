import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import datasets, models, layers, utils


def index_to_char(index: int) -> str:
    return "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[index]


def retrieve_char_data():
    print("Organizing data...")
    data = pd.read_csv(
        "./data/A_Z Handwritten Data.csv",
    ).astype("int")
    letter_labels = data["0"]
    letter_labels += 10
    letter_images = data.drop("0", axis=1)
    letter_images = np.reshape(letter_images, (letter_images.shape[0], 28, 28))
    (digit_images, digit_labels), (digit_images_2, digit_labels_2) = (
        datasets.mnist.load_data()
    )
    images = np.concatenate((digit_images, digit_images_2, letter_images))
    labels = np.concatenate((digit_labels, digit_labels_2, letter_labels))
    return images, labels


def main():
    images, labels = retrieve_char_data()
    images = np.reshape(images, (images.shape[0], 28, 28, 1))
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2
    )

    # Build the neural network model.
    print("Building model...")
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
    model.add(layers.Dense(36, activation="softmax"))

    # Fit the model to the training data
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        train_images, utils.to_categorical(train_labels), epochs=3, batch_size=512
    )
    model.save("./models/characters.keras")

    # Display metrics for the model.
    print("Evaluating network...")
    loss, accuracy = model.evaluate(test_images, utils.to_categorical(test_labels))


if __name__ == "__main__":
    main()
