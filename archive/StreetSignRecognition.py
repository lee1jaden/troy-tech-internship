
#%%
import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D 
from keras.utils.np_utils import to_categorical
import cv2 as cv2


#import the train and test data from mat file to dictionary to numpy array
#data is loaded in two files--one for training, the other for testing
train_data = sio.loadmat('train_32x32.mat')
train_imgs = np.array(train_data['X'])
train_labels = np.array(train_data['y'])
test_data = sio.loadmat('test_32x32.mat')
test_imgs = np.array(test_data['X'])
test_labels = np.array(test_data['y'])
#this function reshapes data so the first dimension distinguishes each image
train_imgs = np.moveaxis(train_imgs, -1, 0) 
test_imgs = np.moveaxis(test_imgs, -1, 0)
#print(train_imgs.shape) #(73257,32,32,3)
#print(test_imgs.shape) #(26032,32,32,3)


#OpenCV preprocessing, change bgr images to grayscale and use Canny to detect edges
test_images = np.empty((test_imgs.shape[0],32,32))
train_images = np.empty((train_imgs.shape[0],32,32))
for i in range (len(train_imgs)):
    gs = np.dot(train_imgs[i], [.2989,.5870,.1140])
    gs = gs.astype(np.uint8)
    train_images[i] = cv2.Canny(gs, 45, 80)
train_images = train_images.reshape((73257,32,32,1))
for i in range (len(test_imgs)):
    gs = np.dot(test_imgs[i], [.2989,.5870,.1140])
    gs = gs.astype(np.uint8)
    test_images[i] = cv2.Canny(gs, 45, 80)
test_images = test_images.reshape((26032,32,32,1))
#plt.imshow(test_images[26031]) #cmap=plt.get_cmap('gray'))


#build the neural network model using convolutional, pooling, and typical neural network layers
network = models.Sequential()
network.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32,32,1)))
network.add(MaxPool2D(pool_size=(2,2), strides=2))
network.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
network.add(MaxPool2D(pool_size=(2,2), strides=2))
network.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='valid'))
network.add(MaxPool2D(pool_size=(2,2), strides=2))
network.add(Flatten())
network.add(Dense(64, activation='relu'))
network.add(Dense(128, activation='relu'))
network.add(Dense(11, activation='softmax'))
network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#train the neural network
#lower batch size, more epochs increase accuracy
#network.fit(train_images, to_categorical(train_labels), epochs=5, batch_size=128)
#network.save_weights('housenumberrecognition(1).h5')
network.load_weights('housenumberrecognition(1).h5')

"""
#performance measurements
test_loss, test_acc = network.evaluate(test_images, to_categorical(test_labels))
print('test_accuracy', test_acc, 'test_loss', test_loss)
"""

#predicting test images and printing with matplotlib
predictions = network.predict(test_images[13040:13056]) #random set of 16 numbers
fig, axes = plt.subplots(4,4)
axes = axes.flatten()
for i, ax in enumerate(axes):
    ax.axis('off')
    ax.imshow(test_images[i+13040], cmap="Greys")
    #ax.imshow(test_imgs[i+13040])
    digit = str(np.argmax(predictions[i]))
    ax.set_title("Prediction: " + digit)
    ax.grid()
print (test_labels[13040:13056])

"""
#trying to put images and canny versions next to each other was unsuccessful
for i in range(int(len(axes)/2)):
    axes[2*i].axis('off')
    axes[2*i].imshow(test_imgs[i])
    digit = str(np.argmax(predictions[i]))
    axes[2*i].set_title("Prediction: " + digit)
    axes[2*i].axis('off')
    axes[2*i+1].imshow(test_images[i], cmap="Greys")
"""

print ("Done")
# %%
