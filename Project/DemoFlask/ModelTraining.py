#!/usr/bin/env python
# coding: utf-8

# ### Bank Note Authentication

# Data were extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.

# In[1]:


##Dataset Link: https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data
import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('BankNote_Authentication.csv')


# In[18]:


df


# In[19]:


### Independent and Dependent features
X=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[5]:


X.head()


# In[6]:


y.head(20)


# In[7]:


### Train Test Split
from sklearn.model_selection import train_test_split


# In[8]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[9]:


### Implement Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)


# In[10]:


## Prediction
y_pred=classifier.predict(X_test)


# In[12]:


### Check Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)


# In[13]:


score


# In[14]:


### Create a Pickle file using serialization 
import pickle
pickle_out = open("classifier.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()


# In[22]:


import numpy as np


# In[18]:


classifier.predict([[2,3,4,1]])


# In[ ]:




