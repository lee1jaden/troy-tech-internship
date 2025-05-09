
#%%

#imported modules
from keras.datasets import mnist
from keras import models
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D 
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import the csv data of preprocessed greyscale images and mnist data
((num_images, num_labels), (num2_images, num2_labels)) = mnist.load_data()
data = pd.read_csv('A_ZHandwrittenData copy.csv')
data.head()


#split into images and labels datasets
text_images = data.copy()
text_labels = text_images.pop(text_images.columns[0])
text_images = np.array(text_images)
text_labels = np.array(text_labels)
text_images = text_images.reshape((text_images.shape[0],28,28))
for i in range (len(text_labels)):
    text_labels[i] = text_labels[i] + 10
all_images = np.concatenate((num_images, num2_images, text_images))
all_labels = np.concatenate((num_labels, num2_labels, text_labels))
print ("Data done...")


#split data into training/test datasets, function also shuffles data (dataset is originally sorted)
train_images, test_images, train_labels, test_labels = train_test_split(all_images, all_labels, test_size=0.20)


#reshape input   train_images.reshape((train_images.shape[0], 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
test_images = test_images.astype('float32') / 255


print ("making model...")
#build the neural network model using convolutional, pooling, and typical neural network layers
network = models.Sequential()
network.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
network.add(MaxPool2D(pool_size=(2,2), strides=2))
network.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
network.add(MaxPool2D(pool_size=(2,2), strides=2))
network.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='valid'))
network.add(MaxPool2D(pool_size=(2,2), strides=2))
network.add(Flatten())
network.add(Dense(64, activation='relu'))
network.add(Dense(128, activation='relu'))
network.add(Dense(36, activation='softmax'))
network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#train the neural networks, lower batch size & more epochs increase accuracy
network.fit(train_images, to_categorical(train_labels), epochs=3, batch_size=512)


#performance measurements
test_loss, test_acc = network.evaluate(test_images, to_categorical(test_labels))
print('test_accuracy', test_acc, 'test_loss', test_loss)
network.save('characterIDmodel.h5')

"""
network = models.load_model('/Users/dracdanne/Desktop/project 3/digitandletternetwork')
labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

predictions = network.predict(test_images[:16])
fig, axes = plt.subplots(4,4)
axes = axes.flatten()
count = 0
for i, ax in enumerate(axes):
    img = test_images[i].reshape((28,28))
    ax.axis("off")
    ax.imshow(img, cmap="Greys")
    letter = labels[np.argmax(predictions[i])] 
    ax.set_title("Prediction: " + letter)
    ax.grid()
    if np.argmax(predictions[i])==test_labels[i]:
        count = count + 1

print (test_labels[:16])
print ("Count:", count)
"""

print ("Done")
# %%
