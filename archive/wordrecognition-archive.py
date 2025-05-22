
#%%

#Kachow
import cv2
import matplotlib.pyplot as plt
import imutils
from imutils.contours import sort_contours
import numpy as np
import keras


#read the colored image data into a bgr numpy array
image = cv2.imread('/Users/dracdanne/Desktop/project 3/mywritingimages/IMG_9720.jpg')

print ('imaging...')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#blur the image to filter out noise (bigger kernel = more blurry)
blurred = cv2.GaussianBlur(gray, (5,5), cv2.BORDER_CONSTANT)
#use thresholding to separate background from foreground
thresh_img = cv2.threshold(blurred, 105, 255, 0)[1]
#convert image to type that Canny can take as input
thresh_img = thresh_img.astype('uint8')
#detect edges and return image showing them
edges = cv2.Canny(thresh_img, threshold1=10, threshold2=250) #20-50 

#make list of contours on the image then sort them left to right
print ('contours...')
contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sort_contours(contours, method="left-to-right")[0]
#draw all the contours on the image as red lines
#cv2.drawContours(image, contours, -1, (0,0,255), 5)
print (str(len(contours))) #45

#initialize lists to hold letter image and box data
chars = []
boxes = []

for cntr in contours:
    (x, y, w, h) = cv2.boundingRect(cntr)

    #make sure contours identified are size of a letter
    if (w>=50 and w<=500) and (h>=110 and h<=1000):

        #draw the rectangle around the character on the image
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 5)
        #extract letter from grayscale image and threshold it
        region = blurred[y:y+h, x:x+w]
        thresh = region
        #thresh = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        #make input image correct shape for classification (28,28)
        (tH, tW) = thresh.shape
        if tH>tW:
            thresh = cv2.resize(thresh, (24, int(24*tW/tH)), interpolation=cv2.INTER_CUBIC)
        else:
            thresh = cv2.resize(thresh, (int(24*tH/tW), 24), interpolation=cv2.INTER_CUBIC)
        dY = int((24 - thresh.shape[0])//2)
        dX = int((24 - thresh.shape[1])//2)
        if dX>dY:
            thresh = cv2.copyMakeBorder(thresh, top=0, bottom=0, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(255,))
        else:
            thresh = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=(255,))
        thresh = cv2.resize(thresh, (24,24))
        thresh = cv2.copyMakeBorder(thresh, top=2, bottom=2, left=2, right=2, borderType=cv2.BORDER_CONSTANT, value=(255,))
        #convert data to type used in neural network
        thresh = thresh.astype('float32')/255.0
        thresh = thresh.reshape((28,28,1))
        #show image for testing
        cv2.imshow('one_letter', thresh) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #add image data and box data to their own lists 
        chars.append(thresh)
        boxes.append((x,y,w,h))

#load the neural network model
print ('networking...')
network = keras.models.load_model('/Users/dracdanne/Desktop/project 3/characterIDmodel.h5')

#change chars list to numpy arrays
chars = np.array(chars) 

#predict characters
print ('predicting...')
labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

"""
predictions = network.predict(chars[:9])
fig, axes = plt.subplots(3,3)
axes = axes.flatten()
count = 0
for i, ax in enumerate(axes):
    img = chars[i].reshape((28,28))
    ax.axis("off")
    ax.imshow(img, cmap="Greys")
    letter = labels[np.argmax(predictions[i])] 
    ax.set_title("Prediction: " + letter)
    ax.grid()
"""

letter_probs = []
predictions = network.predict(chars)
ans = ""
for i in range (len(predictions)):
    letter = labels[np.argmax(predictions[i])]
    letter_probs.append(predictions[i][np.argmax(predictions[i])])
    ans = ans+letter

#print answer and confirm with user for correctness
#print (letter_probs)
print (ans)
cv2.imshow('bounding_box', image)
is_wrong = input('Is this correct?: (Yes/No) ').lower()=="no"
if is_wrong:
    ans = input('Enter the correct word: ').upper()
cv2.waitKey(0)
cv2.destroyAllWindows()

#save the answer
pass


print ('Done')
# %%
