import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import models, layers, utils


def main():
    # Import and organize the data.
    print("Importing data...")
    data = pd.read_csv("./data/A_Z Handwritten Data.csv").astype("int")
    labels = data["0"]
    images = data.drop("0", axis=1)
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
    model.add(layers.Dense(26, activation="softmax"))

    # Fit the model to the training data
    # model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
    # early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
    # history = model.fit(train_X, train_yOHE, epochs=1, callbacks=[reduce_lr, early_stop],  validation_data = (test_X,test_yOHE))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        train_images, utils.to_categorical(train_labels), epochs=3, batch_size=512
    )
    model.summary()
    model.save("./models/letters-cnn.keras")

    # Display metrics for the model.
    print("Evaluating network...")
    loss, accuracy = model.evaluate(test_images, utils.to_categorical(test_labels))


if __name__ == "__main__":
    main()
