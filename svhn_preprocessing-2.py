
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import os
import matplotlib.pyplot as plt
import sys
import pickle
from PIL import Image
from matplotlib.pyplot import imshow
import cv2
import numpy as np
from skimage.color import rgb2gray
import glob
from collections import defaultdict


# In[2]:

test_Struct = pickle.load(open( "test_save.p", "rb" ))


# In[3]:

Struct = pickle.load( open( "save.p", "rb" ) )


# In[4]:

def gen_bbox(structure):
    img_file = defaultdict(list)

    for x in structure:
        img_file[x[0]].append(x[1:])
    
    return img_file

img_file = gen_bbox(Struct)
img_file


# In[ ]:




# In[5]:

test_img_file = gen_bbox(test_Struct)
test_img_file


# In[6]:

image_size = 64
#num_labels = 10
num_channels = 1 # grayscale

def reformat(dataset):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  return dataset


# In[5]:

def conv_channel(img):
    new_array = np.empty(shape=(img.shape[3], img.shape[0],img.shape[1]), dtype=np.float32)
#new_array[9].shape
    for i in range(img.shape[3]):
        img_single = img[:,:,:,i]
        img_single_gray = rgb2gray(img_single)
        new_array[i] = img_single_gray
    return new_array


# In[16]:

conv_channel(reshaped_data).shape


# In[8]:

num_of_data = 13067
def gen_data(files):
    new_array = np.empty(shape=(num_of_data,64,64,1), dtype=np.float32)
    for count, i in enumerate(files.keys()[1:num_of_data]):   #starting index value is 1
        img_path =  os.path.join('train/{}'.format(i))#.format(i)
        img = Image.open(img_path)
    #imshow(img)
        pic = img_file[i]
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
        box = (left_bbox, top_bbox, left_bbox+width_bbox, top_bbox+height_bbox)

        area = img.crop(box).resize([64,64], Image.ANTIALIAS)
    #imshow(area)
        reshaped_data = reformat(rgb2gray(np.asarray(area)))
    #print reshaped_data.shape
        new_array[count] = reshaped_data
    return new_array


new_array = gen_data(img_file)
new_array


# In[9]:

test_new_array = gen_data(test_img_file)


# In[10]:

test_new_array


# In[22]:

cv2.imshow('image',test_new_array[-2])
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[12]:

def gen_labels(folder, files):
    labels_img = []
    for count, i in enumerate(files.keys()[1:num_of_data]):   #starting index value is 1
        img_path =  os.path.join('{}/{}'.format(folder, i))#.format(i)
        img = Image.open(img_path)
    #imshow(img)
        pic = img_file[i]
        labels = [x[0] for x in pic]
    #five_seq_labels = np.empty(shape=(1,5), dtype=np.uint8)
        five_seq_labels = []
    #print labels[0]
        for s in range(5):
            try:
                if labels[s] == 10:
                    labels[s] = 0
                five_seq_labels.append(labels[s])
            #five_seq_labels[count][s] = labels[s]
            except: 
                five_seq_labels.append(10)
            
        labels_img.append(five_seq_labels)
    return np.array(labels_img)
            #five_seq_labels[count][s] = 10
    #five_seq_labels = np.array(five_seq_labels).flatten()
    #five_seq_labels = (np.arange(11) == five_seq_labels[:,None]).astype(np.float32)
    #labels_img.append(five_seq_labels)  
        #five_seq_labels = five_seq_labels.flatten()    
        #five_seq_labels = (np.arange(11) == five_seq_labels[:,None]).astype(np.float32)
#labels_img = gen_labels('train', img_file)
#labels_img
    #labels_img.append(five_seq_labels)
#print labels_img
#labels_img = np.array(labels_img)
#print labels_img.shape
#five_seq_labels
#five_seq_labels[0][:,None]


# In[13]:

labels_test_img = gen_labels('test', test_img_file)
labels_test_img


# In[18]:

def one_hot_en(lbls):
    labels_dataset = []
    for c in range(num_of_data-1):
        labels_dataset.append((np.arange(11) == lbls[c][:,None]).astype(np.float16))
    return labels_dataset

#labels_dataset = one_hot_en(labels_img)
#labels_dataset


# In[19]:

labels_test_dataset = one_hot_en(labels_test_img)
labels_test_dataset


# In[22]:

total_data = pickle.dump(new_array, open("total_data.p", "wb" ) )


# In[23]:

total_data = pickle.load( open( "total_data.p", "rb" ) )


# In[24]:

total_data.shape


# In[15]:

total_label = pickle.dump(np.array(labels_dataset), open("total_labels.p", "wb" ) )


# In[19]:

total_labels = pickle.load( open( "total_labels.p", "rb" ) )


# In[29]:

total_labels[-1]


# In[22]:

Sample_test_data = pickle.dump(test_new_array, open("sample_test_data.p", "wb" ) )


# In[23]:

Sample_test_data = pickle.load( open( "sample_test_data.p", "rb" ) )
Sample_test_data.shape


# In[24]:

Sample_test_label = pickle.dump(np.array(labels_test_dataset), open("sample_test_labels.p", "wb" ) )


# In[26]:

Sample_test_labels = pickle.load( open( "sample_test_labels.p", "rb" ) )
Sample_test_labels

labels
# In[14]:

labels_dataset.shape


# In[100]:

five_seq_labels = np.empty(shape=(1,5), dtype=np.float32)


# In[101]:

five_seq_labels[0][0] = 1


# In[102]:

five_seq_labels = five_seq_labels.flatten()
five_seq_labels


# In[103]:

labels_2 = (np.arange(10) == five_seq_labels[:,None]).astype(np.float32)


# In[28]:

ls = np.array([[6, 10, 10, 10, 10], [6, 10, 10, 10, 10]])


# In[30]:

ls[:,None]


# In[32]:

ls[0]


# In[20]:

labels_test_dataset[-1]


# In[23]:

#concatanate the numpy arrays
total_test_data = pickle.dump(test_new_array, open("total_test_data.p", "wb" ) )


# In[25]:

total_test_data = pickle.load( open( "total_test_data.p", "rb" ) )


# In[26]:

total_test_data.shape


# In[27]:

total_test_label = pickle.dump(np.array(labels_test_dataset), open("total_test_labels.p", "wb" ) )


# In[28]:

total_test_labels = pickle.load( open( "total_test_labels.p", "rb" ) )


# In[29]:

total_test_labels.shape


# In[ ]:



