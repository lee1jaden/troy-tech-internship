#%%
#imported modules
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import the csv data
data = pd.read_csv("A_ZHandwrittenData.csv")
data.head()
#split into images and labels datasets
all_images = data.copy()
all_labels = all_images.pop(all_images.columns[0])
all_images = np.array(all_images)
all_labels = np.array(all_labels)


#split data into training/test datasets, function also shuffles data
train_images, test_images, train_labels, test_labels = train_test_split(all_images, all_labels, test_size=0.25)

#for testing early in coding process
print (train_images.shape)
print (test_images.shape)
print (train_labels.shape)
print (test_labels.shape)


#build the neural network model
network = models.Sequential()
network.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(26, activation='softmax'))
network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#reshape input for neural network
#train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
#test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
"""
other version of normalizing and flattening data
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))
"""

#encode the data
#train_labels = to_categorical(train_labels)
#test_labels = to_categorical(test_labels)

#train the neural network
network.fit(train_images, to_categorical(train_labels), epochs=5, batch_size=128)
#lower batch size, more epochs increase accuracy
"""
network.save_weights('handwrittencharacterrecognition.h5')
network.load_weights('handwrittencharacterrecognition.h5')
"""

#performance measurements
test_loss, test_acc = network.evaluate(test_images, to_categorical(test_labels))
print('test_accuracy', test_acc, 'test_loss', test_loss)

word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}

#predicting test images and printing with matplotlib
predictions = network.predict(test_images[:16])
fig, axes = plt.subplots(4,4)
axes = axes.flatten()
for i, ax in enumerate(axes):
    img = test_images[i].reshape((28,28))
    ax.axis("off")
    ax.imshow(img, cmap="Greys")
    letter = word_dict[np.argmax(predictions[i])] 
    ax.set_title("Prediction: " + letter)
    ax.grid()
print (test_labels[:16])


print ("Done")

#%%
