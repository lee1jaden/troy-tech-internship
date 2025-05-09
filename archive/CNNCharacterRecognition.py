
#imported modules
from keras import models
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D 
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#import the csv data of preprocessed greyscale images
data = pd.read_csv('A_ZHandwrittenData.csv')
data.head()
#split into images and labels datasets
all_images = data.copy()
all_labels = all_images.pop(all_images.columns[0])
all_images = np.array(all_images)
all_labels = np.array(all_labels)

#split data into training/test datasets, function also shuffles data (dataset is originally sorted)
train_images, test_images, train_labels, test_labels = train_test_split(all_images, all_labels, test_size=0.25)

#reshape input for neural network, must be 4-dimensional for Conv2D layer, values are a float from 0 to 1
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
test_images = test_images.astype('float32') / 255
"""
#another way to normalize and flatten data
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5
train_images = train_images.reshape((-1, 784))
test=_images = test_images.reshape((-1, 784))
"""

#for testing early in coding process, has no impact on program
print (train_images.shape)
print (test_images.shape)
print (train_labels.shape)
print (test_labels.shape)

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
network.add(Dense(26, activation='softmax'))
network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#encode the data
#train_labels = to_categorical(train_labels)
#test_labels = to_categorical(test_labels)


#train the neural networks, lower batch size & more epochs increase accuracy
network.fit(train_images, to_categorical(train_labels), epochs=3, batch_size=512)
"""
network.save_weights('convolutioncharacterrecognition.h5')
network.load_weights('convolutioncharacterrecognition.h5')
"""

#performance measurements
test_loss, test_acc = network.evaluate(test_images, to_categorical(test_labels))
print('test_accuracy', test_acc, 'test_loss', test_loss)

word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}

"""
#predicting test images
predictions = network.predict(test_images[:50]) #tests first 50 images of test dataset
print (np.argmax(predictions, axis=1)) #takes index of largest argument of output from last Dense layer
print (test_labels[:50])
"""


print ("Done")