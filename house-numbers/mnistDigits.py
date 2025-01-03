
#imported modules
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical
import numpy as np


#set up training/test data 
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#make the neural network model
network = models.Sequential()
network.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

"""
#reshape input for neural network
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
"""
#victorzhou version of normalizing and flattening data
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))


#encode the data
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#train the neural network
network.fit(train_images, train_labels, epochs=3, batch_size=256)
#victorzhou used batch_size=32, lower batch size increased accuracy


#performance measurements
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc', test_acc, 'test_loss', test_loss)

"""
#predicting test images
predictions = network.predict(test_images[:5])
print (np.argmax(predictions, axis=1))
print (test_labels[:5])
"""