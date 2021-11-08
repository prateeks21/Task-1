#!/usr/bin/env python
# coding: utf-8

# # Author: Prateek Soni

# # GRIP @ The Sparks Foundation

# In this regression task I have predicted the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression problem as it has just one predictor.
# 
# 

# # Step:1 Importing Useful Python Library

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Step:2 Importing Dataset

# In[2]:


data=pd.read_csv("data.csv")
print('Importing Data Successfully')


# In[3]:


print('First ten data')
data.head(10)


# Preparing Data for Machine learning

# In[25]:


data.info()


# In[4]:


#Data cleaning
data.isnull().sum()


# # Step:3 Data Visualisation

# In[5]:


x=np.array(data[['Hours']])


# In[6]:


y=np.array(data[["Scores"]])


# In[7]:


plt.scatter(x,y)
plt.title("Hours vs Percentage")
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# In[8]:


print('We can see that Scores increses as the no. of hours studied is increase')
print('hence we can conclude that there exist a positive linear relation between the number of hours studied and percentage of score.')


# # Step:4 Train-Test-Split

# In[9]:


x=data[['Hours']]
y=data[['Scores']]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2)


# # Step:5 Training Algorithm

# In[10]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(x_train, y_train)
print('Training Complete')


# # Step:6 Plotting the Line of Regression

# In[11]:


regressor.coef_


# In[12]:


regressor.intercept_


# In[13]:


x_test


# In[21]:


y_test


# In[27]:


y_pred=regressor.predict(x_test)
y_pred


# In[28]:


pd.DataFrame(np.c_[x_test,y_test,y_pred], columns=['Study Hours', 'Original Student Marks','Predicted Student Marks'])


# In[29]:


regressor.score(x_test,y_test)


# In[30]:


plt.scatter(x_test,y_test)
plt.plot(x_train, regressor.predict(x_train), color='red')


# In[34]:


#Predcting the 'Marks' with the given value of 'Hours'
regressor.predict([[9.25]])


# # Step:7 Evaluating the model

# In[48]:


from sklearn import metrics
print('Mean Absolute Error', metrics.mean_absolute_error(y_test,y_pred))


# # Conclusion

#  I have carried out the prediction using Supervised Machine Leaning and evaluated performance of the model.
#  From above analysis, I reached to the conclusion that  if a student study for 9.25 then he/she will score 90.89 percentage/marks.

# In[ ]:





# In[ ]:




