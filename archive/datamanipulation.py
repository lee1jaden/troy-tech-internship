#Manipulating data from folders to images to csv file
#%%

import os
import matplotlib.pyplot as plt
import shutil
import numpy as np
import pandas as pd
import random
import cv2



""
# this program converts all the images into one single csv file that can be quickly read
# 
#

#construct an empty numpy array
array = np.empty((115320,150*500))
end_dir = '/Users/dracdanne/Desktop/project 3/allwordsimages'
image_list = os.listdir(end_dir)
#random.shuffle(image_list)
#num = random.randint(0,115319)
#num = 107895
#print(num)

#go through the list of images and add each to the numpy array
for i in range (len(image_list)):
    #read image from path/directory
    img = cv2.imread(os.path.join(end_dir, image_list[i]))
    #convert to a grayscale image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #if the image is bigger in either direction than 150 by 500, resize with same aspect ratio to fit in that size
    if img.shape[1]>500:
        aspect_ratio = img.shape[0]/img.shape[1]
        img = cv2.resize(img, dsize=(500, int(500*aspect_ratio)), interpolation=cv2.INTER_CUBIC)
    if img.shape[0]>150:
        aspect_ratio = img.shape[0]/img.shape[1]
        img = cv2.resize(img, dsize=(int(150/aspect_ratio), 150), interpolation=cv2.INTER_CUBIC)
    #add background color to make each image the same shape
    right_space = 500-img.shape[1]
    bot_space = 150-img.shape[0]
    img = cv2.copyMakeBorder(img, top=0, bottom=bot_space, left=0, right=right_space, borderType=cv2.BORDER_CONSTANT, value=(255,))
    img = img.flatten()
    #array = np.append(array, img, axis=0)
    #assign image data to numpy array
    array[i] = img
    #plt.imshow(array[i].reshape((150,500)), cmap="Greys")
    print (i)

#save the numpy array to a csv file
pd.DataFrame(array).to_csv("wordimages.csv")
""



"""
# this program finds the maximum height and width
# of the images in the dataset

max_height = 0 #277
max_width = 0 #1934
end_dir = '/Users/dracdanne/Desktop/project 3/allwordsimages'
img_list = os.listdir(end_dir)
#random.shuffle(img_list)
maxw_index = 0
for i in range (100):
    if img_list[i].find('.png')>0:
        image = plt.imread(os.path.join(end_dir, img_list[i]))
        image = np.array(image)
        #plt.imshow(image, cmap='binary')
        if image.shape[1]>max_width:
            maxw_index = i
        max_height = max(max_height, image.shape[0])
        max_width = max(max_width, image.shape[1])
        print (i)
        #print (img_list[i])
print (max_height)
print (max_width)
print (maxw_index)
"""



"""
# this program moves all the files from each folder of the 'words'
# folder into one folder called 'allwordsimages'
count_files = 0
ds_store_num = 0
end_dir = '/Users/dracdanne/Desktop/project 3/allwordsimages'
words_path = '/Users/dracdanne/Desktop/project 3/words'
words_folders = os.listdir(words_path)
words_folders.remove('.DS_Store')
words_folders.sort()
#print (len(words_folders))
for i in range(len(words_folders)):
    folders_of_images = os.listdir(os.path.join(words_path, words_folders[i]))
    if folders_of_images.count('.DS_Store')!=0:
            ds_store_num = ds_store_num + folders_of_images.count('.DS_Store')
            folders_of_images.remove('.DS_Store')
    #folders_of_images.sort()
    for j in range(len(folders_of_images)):
        images = os.listdir(os.path.join(words_path, words_folders[i], folders_of_images[j]))
        if images.count('.DS_Store')!=0:
            ds_store_num = ds_store_num + images.count('.DS_Store')
            images.remove('.DS_Store')
        #images.sort()
        count_files = count_files + len(images)
        for f in images:
            shutil.copy(os.path.join(words_path, words_folders[i], folders_of_images[j], f), end_dir)
        #print(count_files)

print (count_files)
print (ds_store_num)
"""



"""
# reading the letter.data file and printing the images
# didn't end up using the letter.data file since doesn't have words
file = pd.read_csv('letter.data', sep="\t")
#print (file)
file  = np.array(file)
img = file[0][6:134]
img = img.astype('float32') / 255
img = img.reshape((16,8))
plt.axis("off")
plt.imshow(img, cmap="Greys")
print (file[0][1])
"""

print("Done")
# %%


