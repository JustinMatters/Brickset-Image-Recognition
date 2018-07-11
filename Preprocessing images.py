
# coding: utf-8

# In[4]:


#!/usr/bin/python
import os
import sys
import random
from PIL import Image # used for image resizing
import pickle # used for retrieving saved data
import shutil # used for rapid image copying
import collections # used for checking category sizes


# In[5]:


# resize the images in a specified path and write them back to a different path

# this is slightly larger than the 224x224 dimensions needed for VGG16 which allows for data augmentation later if required
width = 256
height = 256

in_path = "C:/Users/Justin/Pictures/Lego/thumbnails/"
out_path = "C:/Users/Justin/Pictures/Lego/preprocessed/"
os.chdir(in_path)
contents = os.listdir(in_path)

def resize():
    counter = 0
    for item in contents:
        # check that the file in question is a file not a folder
        if os.path.isfile(in_path+item):
            counter +=1
            im = Image.open(in_path+item)
            imResize = im.resize((width,height), Image.ANTIALIAS)
            imResize.save(out_path + item, 'JPEG', quality=90)
    # tell us how many images we managed to resize
    print (counter)
        
resize()


# In[6]:


# Getting back the metadata for the images
path = "C:/Users/Justin/Pictures/Lego/"
os.chdir(path)
with open('LegoDataClean.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    imported_dataset = pickle.load(f)

# check data is what we expect
print (len(imported_dataset))
print (imported_dataset[30:35])


# In[7]:


#some of the classes may prove to be very small, we need to  find the large sets with > 100 images

# create a dictionary of categories and number of members
categories_extracted = [x[2] for x in imported_dataset]
counted_categories =collections.Counter(categories_extracted)
print (counted_categories)


# In[8]:


# shuffle up our examples to ensure that train, validation and test sets have similar distributions
random.shuffle(imported_dataset)

# then sort by set type
set_type_dataset = sorted(imported_dataset, key=lambda x: x[2])

print (set_type_dataset[30:35])


# In[9]:


# create a new training set list which only contains items in a subsetted list
limited_list = ['Duplo', 'Star', 'City', 'Creator', 'Bionicle', 'Ninjago', 'Town',
                'Racers', 'Technic', 'Castle', 'System', 'LEGOLAND', 'Space', 'Sports', 
                'Explore', 'Trains', 'Fabuland', 'HERO', 'Marvel', 'DC']


# In[10]:


#create a new list which only contains entries from our chosen categories
limited_dataset = []

for row in set_type_dataset:
    if row[2] in limited_list:
        limited_dataset += [row]

# check our new dataset looks plausible
print (limited_dataset[30:35])
print (len(limited_dataset))


# In[11]:


# set up our keras folders

# specify base directories
source_directory = "C:/Users/Justin/Pictures/Lego/preprocessed/"
train_directory = "C:/Users/Justin/Pictures/Lego/data/train/"
validation_directory = "C:/Users/Justin/Pictures/Lego/data/validation/"
test_directory = "C:/Users/Justin/Pictures/Lego/data/test/"

# need to extract the categories to set up appropriate folders in train, test and validate
categories = []
for row in limited_dataset:
    categories += [row[2]]

categories = list(set(categories))
# check our list is correct
print (categories)

for category in categories:
    os.makedirs(os.path.dirname(train_directory+category+"/"), exist_ok=True)
    os.makedirs(os.path.dirname(test_directory+category+"/"), exist_ok=True)
    os.makedirs(os.path.dirname(validation_directory+category+"/"), exist_ok=True)


# In[12]:


# load the various classes of images and place them in train validation and test folders in an 8:1:1 ratio
import shutil 
row_number = 0

def assign_image(image_name, source, target):
    shutil.copyfile(source + image_name, target + image_name)
    #image = Image.open(source_directory+image_name)
    #image.save(target_directory + image_name, 'JPEG', quality=90)

# cycle over dataset rows
for row in limited_dataset:
    # assign 1/10 to validations, 1/10 to test and the rest to train
    if row_number % 10 == 0:
        # note generators for filename AND also for destination directory
        assign_image(row[0] + "_" + row[2] + "_" + row[4] + '.jpg', source_directory, validation_directory+row[2]+"/")
    elif (row_number - 1) % 10 == 0:
        assign_image(row[0] + "_" + row[2] + "_" + row[4] + '.jpg', source_directory, test_directory+row[2]+"/")
    else:
        assign_image(row[0] + "_" + row[2] + "_" + row[4] + '.jpg', source_directory, train_directory+row[2]+"/")
    row_number += 1


# In[13]:


# finally lets pickle the subset we intend to use in case it comes in handy
import pickle

path = "C:/Users/Justin/Pictures/Lego/"
os.chdir(path)

with open('LegoDataTop20Categories.pkl', 'wb') as file:  # Python 3: open(..., 'wb')
    pickle.dump(limited_dataset, file)

