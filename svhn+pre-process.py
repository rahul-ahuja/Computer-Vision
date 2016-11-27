
# coding: utf-8

# In[1]:

import sys
import pickle
import digitStruct


# In[2]:

get_ipython().magic(u'matplotlib inline')

import matplotlib.pyplot as plt


# In[9]:

#fileCount = 0
#for dsObj in digitStruct.yieldNextDigitStruct('train/digitStruct.mat'):
#    fileCount += 1
#    for bbox in dsObj.bboxList[0:10]:
        #csvLine = "{},{},{},{},{},{}\n".format(
#        print dsObj.name
                #bbox.label, bbox.left, bbox.top, bbox.width, bbox.height)
            #csvFile.write(csvLine)
    #print("Number of image files: {}".format(fileCount))


# In[11]:

# Extracting the bounding box for the training data
fileCount = 0
data = []
for dsObj in digitStruct.yieldNextDigitStruct('train/digitStruct.mat'):
    fileCount += 1
    for bbox in dsObj.bboxList:
        data.append([dsObj.name,bbox.label, bbox.left, bbox.top, bbox.width, bbox.height])

print data


# In[14]:

# Extracting the bounding box for the test data
fileCount = 0
test_data = []
for dsObj in digitStruct.yieldNextDigitStruct('test/digitStruct.mat'):
    fileCount += 1
    for bbox in dsObj.bboxList:
        #csvLine = "{},{},{},{},{},{}\n".format(
        test_data.append([dsObj.name,bbox.label, bbox.left, bbox.top, bbox.width, bbox.height])

print test_data


# In[ ]:




# In[57]:

#Struct = pickle.dump( data, open( "save.p", "wb" ) )


# In[3]:

Struct = pickle.load( open( "save.p", "rb" ) )


# In[7]:

Struct
#test_Struct = pickle.dump(test_data, open( "test_save.p", "wb" ) )


# In[16]:

test_Struct = pickle.load( open( "test_save.p", "rb" ) )
test_Struct


# In[ ]:




# In[8]:

# appending image file names to the bounding box details
from collections import defaultdict
img_file = defaultdict(list)

for x in Struct:
    img_file[x[0]].append(x[1:])
    
img_file


#len(Struct)


# In[10]:


    


# In[21]:

digits = {}
for k in labels_digit:
    if k in digits:
        digits[k] = digits[k] + 1
    else:
        digits[k] = 1

digits[0] = digits[10]
del digits[10]
digits
    


# In[22]:

#plot for histogram of digit distribution

plt.bar(range(len(digits)), digits.values(), align='center')
plt.xticks(range(len(digits)), digits.keys())

plt.show()


# In[15]:

diction = {}

diction['1'] = []


# In[19]:




# In[33]:

train_info = Counter(Counter(l).values())


# In[36]:

plt.figure(figsize=(16,10), dpi=75)
plt.bar(map(lambda n: n-0.4, train_info.keys()), train_info.values())
#plt.bar(map(lambda n: n-0.4, test_info.keys()), test_info.values())
#plt.bar(map(lambda n: n-0.4, extra_info.keys()), extra_info.values())
plt.title('number of different labels in train data')
# plt.legend(['expected net rewards'])
plt.xlabel('length of labels')
plt.ylabel('quantity of each lable')
plt.show()


# In[29]:

from PIL import Image
from matplotlib.pyplot import imshow
import numpy as np

img = Image.open('train/32422.png')
imshow(img)
#np.asarray(img).shape
#left = 246
#top = 77
#width = 160
#height = 219

#box = (left, top, left+width, top+height)

#area = img.crop(box)
#imshow(area)


# In[ ]:




# In[47]:

left = 323
top = 81
width = 50
height = 219
box = (left, top, left+width, top+height)

#[9, 323, 81, 96, 219]
area = img.crop(box)
imshow(area)


# In[98]:

#height remains the same, left comes from first digit, top should be the smallest, width should be added and then subtracted by small percent


# In[127]:

#22.png [5, 24, 4, 10, 16]
#22.png [1, 34, 5, 7, 16]
#22.png [5, 40, 5, 11, 16]
#imagename, label, left, top, width, height


#329.png [1, 17, 4, 5, 14]
#329.png [1, 21, 4, 6, 14]
#329.png [4, 27, 4, 10, 14]

img = Image.open('train/329.png')
imshow(img)
left = 17 
top = 4
width = 20
height = 14

#left = 246
#width = 120
#top = 90
box = (left, top, left+width, top+height)

area = img.crop(box)
imshow(area)


# In[54]:

#pic = img_file['10.png']
#[x[0] for x in pic]
pic = img_file['3242.png']


# In[55]:

labels = [x[0] for x in pic]
left = [x[1] for x in pic]
top = [x[2] for x in pic]
width = [x[3] for x in pic]
height = [x[4] for x in pic]


# In[56]:

left[0]


# In[31]:

five_seq_labels = []
for s in range(5):
    try:
        if labels[s] == 10:
            labels[s] = 0
        five_seq_labels.append(labels[s])
    except: 
        five_seq_labels.append(10)
        
five_seq_labels


# In[59]:

img = Image.open('train/3242.png')
#imshow(img)
left_bbox = left[0] #+ 0.25 * left[0]
top_bbox = top[0] #+ 0.25 * top[0]
width_bbox = sum(width)
height_bbox = height[0]# + 0.25 * height[0]

#left = 246
#width = 120
#top = 90
box = (left_bbox, top_bbox, left_bbox+width_bbox, top_bbox+height_bbox)

area = img.crop(box).resize([64,64], Image.ANTIALIAS)
imshow(area)


# In[39]:

import cv2
import numpy as np
from skimage.color import rgb2gray
#from skimage.io import imread
#from scipy.misc import imread
#im = imread(area)
#imshow(rgb2gray(np.asarray(area)))
#(rgb2gray(np.asarray(img))).shape
#im = imread(area)
print (rgb2gray(np.asarray(area))).shape
#imshow(rgb2gray(np.asarray(area)))

image_size = 64
num_labels = 10
num_channels = 1 # grayscale

def reformat(dataset):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  return dataset

reshaped_data = reformat(rgb2gray(np.asarray(area)))



#imshow(reshaped_data[0])#.shape
#cv2.imshow('image',reshaped_data[0])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#imshow(reshaped_data[0])
#imshow(img)


# In[7]:

import glob
dfiles = glob.glob('train/*.png')


# In[9]:

len(dfiles)


# In[38]:

import os
import numpy as np
data = []
labels_img = []
for i in img_file.keys()[1:500]:   #starting index value is 1
    img_path =  os.path.join('train/{}'.format(i))#.format(i)
    img = Image.open(img_path)
    #imshow(img)
    pic = img_file[i]
    #print pic
    #print img
    #imshow(img)
    #imshow(rgb2gray(np.asarray(img)))
    #print img_file[i]
    labels = [x[0] for x in pic]
    left = [x[1] for x in pic]
    top = [x[2] for x in pic]
    width = [x[3] for x in pic]
    height = [x[4] for x in pic]
    #print labels
    left_bbox = left[0] 
    top_bbox = top[0]
    width_bbox = sum(width)
    height_bbox = height[0]

#left = 246
#width = 120
#top = 90
    box = (left_bbox, top_bbox, left_bbox+width_bbox, top_bbox+height_bbox)

    area = img.crop(box)
    #data
    data.append(np.array(rgb2gray(np.asarray(area)), dtype = np.float16))
    five_seq_labels = []
    for s in range(5):
        try:
            if labels[s] == 10:
                labels[s] = 0
            five_seq_labels.append(labels[s])
        except: 
            five_seq_labels.append(10)
            
    labels_img.append(five_seq_labels)
    #print i
    #imshow(area)
    #left_bbox = left[0] 
    #top_bbox = top[0]
    #width_bbox = sum(width)
    #height_bbox = height[0]
#print data
print labels_img
print "Length of Data is {}".format(len(data))
print "Length of labels in the data is {}".format(len(labels_img))
imshow(data[2])
#left = 246
#width = 120
#top = 90
    #box = (left_bbox, top_bbox, left_bbox+width_bbox, top_bbox+height_bbox)

    #area = img.crop(box)

#five_seq_labels = []
#for s in range(5):
#    try:
#        five_seq_labels.append(labels[s])
#    except: 
#        five_seq_labels.append(10)
        
#five_seq_labels


# In[39]:

imshow(data[2])


# In[1]:

#for i in dfiles[0:100]:
#    img = Image.open('train/10.png').format{i}
#    imshow(img)
#    left_bbox = left[0] 
#    top_bbox = top[0]
#    width_bbox = sum(width)
#    height_bbox = height[0]

#left = 246
#width = 120
#top = 90
#    box = (left_bbox, top_bbox, left_bbox+width_bbox, top_bbox+height_bbox)

#    area = img.crop(box)
#    imshow(area)


# In[ ]:




# In[ ]:



