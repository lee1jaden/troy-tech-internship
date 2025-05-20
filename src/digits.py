from keras.datasets import mnist
from keras import models, layers, utils


def index_to_digit(index: int) -> str:
    return str(index)


def main():
    # Import and organize the data.
    print("Importing data...")
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Build and fit the neural network model.
    print("Building model...")
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(784, activation="relu"))
    model.add(layers.Dense(784, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        train_images, utils.to_categorical(train_labels), epochs=5, batch_size=128
    )
    model.summary()
    model.save("./models/digits.keras")

    # Display metrics for the model.
    print("Evaluating network...")
    loss, accuracy = model.evaluate(test_images, utils.to_categorical(test_labels))


if __name__ == "__main__":
    main()
