
# coding: utf-8

# In[2]:


#import the libraries we will need

import urllib
from bs4 import BeautifulSoup as soup 
import os

path = "C:/Users/Justin/Pictures/Lego"
baseURL = "https://brickset.com/sets/random/"
baseURL = "https://brickset.com/browse/sets"

# initialise the program path
os.chdir(path)


# In[4]:


#create a list to hold our return information
soup_list = []

# open with urrllib (note this form is python 3 specific)
req = urllib.request.Request(baseURL)
opened = urllib.request.urlopen(req)
page_HTML = opened.read()
opened.close()
# convert HTML to a soup object for parsing
soup_list += [soup(page_HTML, "html.parser")]
    
print (len(soup_list))


# In[6]:


# we can examine the text
print (soup_list[0].text)
# we can examine the html
print (soup_list[0].prettify())


# In[5]:


# extract themes
theme_links = []
# find link blocks
for link in soup_list[0].find_all('a'):
    # extract the relative link
    link_text = link.get('href')
    # reject extraneous links on  page non theme links and theme links specific to 2018
    if (link_text[:11] == "/sets/theme") and (link_text[-9:-5] != "year"):
        theme_links += ["https://brickset.com"+link_text] #store the full URL not just hte relative
        #print(link.find('img')['src'])
        
# check that the output is as desired
print(len(theme_links))
print(theme_links[0:10])


# In[6]:


#create a list to hold our return information
theme_soup_list = []

# open and copy the desired pages
for theme in theme_links[0:10]: # can insert numbers into [0:] during testing to shorten run times
    # open with urrllib (note different in python 2 and 3)
    req = urllib.request.Request(theme)
    opened = urllib.request.urlopen(req)
    page_HTML = opened.read()
    opened.close()
    # convert HTML to a soup object for parsing
    theme_soup_list += [soup(page_HTML, "html.parser")]
    
print (len(theme_soup_list))


# In[9]:


# the set_soup_list url only retrieves the first 25 matches
# lets find out how many results there are in each set and create a more complete list
complete_URL_list = []
for i in range(len(theme_soup_list)): # need to use iterator so we can also iterate over theme_links
    # we need to extract the number of matches in the current group
    matches_text = (theme_soup_list[i].find('div', class_='results').text)
    matches_list = matches_text.split()
    matches = int(matches_list[4])
    # need pages to be two higher than the actual number to account for python for loops and partial pages
    pages = (int(matches/25)+2) 
    # not lets create a new more complete list of pagers to spider
    complete_URL_list += [theme_links[i]]
    for j in range(2,pages):
        complete_URL_list += [theme_links[i]+"/page-" +str(j)]
        
print (len(complete_URL_list))
print (complete_URL_list[0:10])


# In[10]:


# roll up extraction of thumbnail address, main link, name and metadata list into one loop

def getSoup (targetURL):
    req = urllib.request.Request(targetURL)
    opened = urllib.request.urlopen(req)
    page_HTML = opened.read()
    opened.close()
    # convert HTML to a soup object for parsing
    #print("got soup")
    return soup(page_HTML, "html.parser")

def processSoup (soup_in):
    processed_tuple_list = []
    matches = (soup_in.find_all('article', class_='set'))
    #print(len(matches))
    for match in matches[0:]:
        # get the image url for the set
        set_thumb_URL = match.find('img')['src']
        # get the title and split off the set name
        title = match.find('img')['title']
        set_name = title.split(':')[1].strip()
        # get the tags for the set
        tag_soups = match.find_all('div', class_='tags')
        primary_tags = tag_soups[0].text.strip().split()
        set_number = primary_tags[0] # first primary tag gives more detail than the split of the name
        set_type = primary_tags[1]
        year = primary_tags[-1] # always the last primary tag
        secondary_tags = tag_soups[-1].text.strip().split("  ")
        # make a tuple of the set_number, set_name, set_type, set_thumb_URL, year, secondary_tags
        item_tuple = (set_number, set_name, set_type, set_thumb_URL, year, secondary_tags)
        processed_tuple_list += [item_tuple]
    return processed_tuple_list
    
#cycle over all the scraped pages however do not  preserve the soups as that would get huge, iterate over them instead
lego_set_tuple_list = []
for page_URL in complete_URL_list[0:]:
    soup_to_process = getSoup(page_URL)
    lego_set_tuple_list += processSoup(soup_to_process)

# check that the output is as desired  
print (len(lego_set_tuple_list))
print (lego_set_tuple_list[5:10])


# In[11]:


# Save our massive file to disk so we don't lose it :-)
import csv

path = "C:/Users/Justin/Pictures/Lego/"
os.chdir(path)

outfile=open('Lego Data2.csv','w')
writer=csv.writer(outfile)
writer.writerow(['setNumber', 'setName', 'setType','setThumbURL', 'year', 'secondaryTags'])
writer.writerows(lego_set_tuple_list)


# In[12]:


# we can use pickle to save a more readily python readable form of our data
import pickle

path = "C:/Users/Justin/Pictures/Lego/"
os.chdir(path)

with open('LegoData2.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(lego_set_tuple_list, f)


# In[15]:


# Getting back the objects:
path = "C:/Users/Justin/Pictures/Lego/"
os.chdir(path)
with open('LegoData2.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    spare_copy = pickle.load(f)
    
print (len(spare_copy))
print (spare_copy[30:35])


# In[16]:


# check for duplicates
list_spare = [i[0] for i in spare_copy]
set_spare = set(list_spare)
print(len(set_spare))
# acceptably small number of duplicates (about 50 out of 15000 = 0.4%)


# In[17]:


# create a cleansed set list
intermediate_set_list = []
cleansed_set_list = []
#first remove any lines where the URL is not a .jpg 
for lego_set in lego_set_tuple_list[0:]:
    #print (lego_set[3][-17:-13])
    if lego_set[3][-17:-13] == '.jpg':
        intermediate_set_list += [lego_set]
        
print (len(intermediate_set_list))

# remove duplicate entries
for lego_set in intermediate_set_list:
    if lego_set[0] in set_spare:
        set_spare.remove(lego_set[0])
        cleansed_set_list += [lego_set]
        
print (len(cleansed_set_list))


# In[18]:


# download thethumbnails
path = "C:/Users/Justin/Pictures/Lego/thumbnails/"
os.chdir(path)
for lego_set in cleansed_set_list[0:5]:  # can insert numbers into [0:] during testing to shorten run times
    #open and name file
    imagefile = open(lego_set[0] + "_" + lego_set[2] + "_" + lego_set[4] + '.jpg', "wb")
    #open url and write to file
    imagefile.write(urllib.request.urlopen(lego_set[3]).read())
    # close file
    imagefile.close()


# In[87]:


# finally lets pickle the cleaned data
import pickle

path = "C:/Users/Justin/Pictures/Lego/"
os.chdir(path)

with open('LegoDataClean.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(cleansed_set_list, f)


# In[88]:


# Getting back the objects:
path = "C:/Users/Justin/Pictures/Lego/"
os.chdir(path)
with open('LegoDataClean.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    checkCopy = pickle.load(f)
    
print (len(checkCopy))
print (checkCopy[30:35])
print("Done")


# In[ ]:


#ADDENDA
# declaring a user agent to disuade the site from bouncing us
user_agent = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36'
headers = { 'User-Agent' : user_agent }

# open and copy the desired pages
# open with urrllib (note this form is python 3 specific)
req = urllib.request.Request(baseURL), None, headers)

