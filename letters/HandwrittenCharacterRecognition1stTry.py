# Numpy, openCV, Keras, Tensorflow (Keras uses TensorFlow in 
# backend and for some image preprocessing) , Matplotlib and Pandas 
# were used to achieve the goal of handwritten character recognition. 
from keras.datasets import mnist #dataset of handwritten digits
import matplotlib.pylot as plt #used for plotting data
import cv2 #image processing
import numpy as np #better data structures
from keras.models import Sequential #neural network model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
import pandas as pd #data manipulation
import numpy as np
from sklearn.model_selection import train_test_split #splits data into train/test sets
from keras.utils import np_utils
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook #???????
from sklearn.utils import shuffle #shuffle train/test data


#read the data
images = pd.read_csv('A_Z Handwritten Data.csv')
print (images.head(5))


#split data into images and labels (move labels from column to own ndarray)
pass


#assigns the train and test arrays from x and y
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2)
#reshapes arrays to (#num dimensions, 28, 28) so can display as image
train_x = np.reshape(train_x.values, (train_x.shape[0], 28,28))
test_x = np.reshape(test_x.values, (test_x.shape[0], 28,28))
print("Train data shape: ", train_x.shape)
print("Test data shape: ", test_x.shape)

word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}


#plotting the number of alphabets in the dataset (matching letters to number of occurrences based on labels)
pass


#Shuffling the data ... (to display thresholded images)
shuff = shuffle(train_x[:100])

fig, ax = plt.subplots(3,3,figsize = (10,10))
axes = ax.flatten()

for i in range(9):
    pass
    axes[i].imshow(np.reshape(shuff[i], (28,28), cmap="Greys")
plt.show()


#reshape training and test datasets to put in model ( )
pass


# making the neural network using the training dataset
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='valid'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))

model.add(Flatten())

model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))

model.add(Dense(26, activation="softmax"))
        

#Making model predictions based off the trained network

pred = model.predict(test_X[:9])
print(test_X.shape)


#more about predictions
fig, axes = plt.subplots(8,8,figsize=(15,18))
axes = axes.flatten()

for i,ax in enumerate(axes):
    img = np.reshape(test_X[i], (28,28))
    ax.imshow(img, cmap="Greys")
    pred = word_dict[np.argmax(test_yOHE[i])]
    ax.set_title("Prediction: " + pred)
    ax.grid()




