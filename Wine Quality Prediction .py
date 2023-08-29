#!/usr/bin/env python
# coding: utf-8

# In[52]:


#importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[53]:


#dataset

wine = "winequality-red.csv"


# In[54]:


data = pd.read_csv(wine)


# # Given Data

# In[55]:


data


# In[56]:


#shape of the data

data.shape


# In[57]:


#data information

data.info


# In[58]:


#checking nulls

data.isnull().sum()


# In[59]:


#data distribution

data.head(6)


# In[60]:


data.tail()


# # Data Analysis 
# 
# 
# 
# data.describe()

# # GRAPHS

# In[61]:


sns.catplot(x='quality', data= data, kind='count')


# In[62]:


plot = plt.figure(figsize =(5,5))
sns.barplot(x='quality', y='fixed acidity',data=data)
plt.show()


# In[63]:


plot = plt.figure(figsize =(5,5))
sns.barplot(x='quality', y='density',data=data)
plt.show()


# In[64]:


plot = plt.figure(figsize =(5,5))
sns.barplot(x='quality', y="residual sugar",data=data)
plt.show()


# In[65]:


##Heatmaps in seaborn

correlation = data.corr()


# In[66]:


plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar= True, square = True, fmt= ".1f", annot = True, annot_kws={"size":8}, cmap = "Blues" )


# # Data pre-processing

# In[67]:


#taking x and y


# In[68]:


x = data.drop("quality", axis =1)


# In[69]:


x


# In[70]:


y = data["quality"].apply(lambda y_values:1 if y_values>=7 else 0)


# In[71]:


y


# In[72]:


#data spliting 

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=3)


# In[73]:


x_train.shape


# In[74]:


x_test.shape


# In[75]:


y_train.shape


# In[76]:


y_test.shape


# # RANDOM FOREST CALLISIFIER MODEL

# In[77]:


model = RandomForestClassifier()


# In[78]:


model.fit(x_train,y_train)


# In[79]:


x_test_prediction = model.predict(x_test)


# In[80]:


test_data_accuracy = accuracy_score(x_test_prediction,y_test)


# In[81]:


test_data_accuracy = accuracy_score(x_test_prediction,y_test)


# In[82]:


print('data accuracy', test_data_accuracy)


# # Predictive System

# In[83]:


input_data = (7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0)
array = np.asarray(input_data)
reshape = array.reshape(1,-1)
predictions = model.predict(reshape)
print (predictions)
if predictions[0]==1:
    print("good quality wine")
else:
    print("Bad quality wine")


# In[84]:


#system is ready


# THANK YOU..

# By Shreya Ghosal

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




