#!/usr/bin/env python
# coding: utf-8

# # Predict Boston House Price Using Python & Linear Regression

# In[1]:


# Import library
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Load the Boston Housing Data set from sklearn.datasets 
from sklearn.datasets import load_boston
boston=load_boston()


# In[3]:


data = boston.data
data=pd.DataFrame (data =data, columns= boston.feature_names)
data.head()


# In[4]:


data['Price']= boston.target
data.head()


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# # Data Visualization

# In[7]:


sns.pairplot(data)


# # Review the shape of distribution

# In[8]:


rows =2
cols =7

fig, ax = plt.subplots(nrows =rows, ncols =cols, figsize = (16,4))
col = data.columns
index = 0
for i in range(rows):
    for j in range(cols):
        sns.distplot(data[col[index]], ax =ax[i][j])
        index = index + 1
plt.tight_layout()


# If the features are sckewed then it will not predict the price (target variable).
# (e.g: Crime, ZN, CHAS, Age, B)
# 
# If the features are nomally distributed, then the feature might serve as an important role to predict the target variable.
# (e.g: RM, LSAT, PTRATIO)
# 

# # Feature selection

# In[9]:


corrmat = data.corr()
print(corrmat)

mask = np.array(corrmat)
mask[np.tril_indices_from(mask)] = False


# In[10]:


fig, ax = plt.subplots()
fig.set_size_inches(10,8)
sns.heatmap(corrmat, mask=mask,vmax=0.8, square=True,annot=True)


# In[11]:


corrmat.index.values


# In[12]:


# Select highly correlated features with target variable

def getCorrelatedfeature (corrdata, threshold):
    feature = []
    value = []
    
    for i, index in enumerate(corrdata.index):
        if abs(corrdata[index]) > threshold:
            feature.append(index)
            value.append(corrdata[index])

    df = pd.DataFrame(data = value, index = feature, columns = ['Corr Value'])
    return df


# In[13]:


threshold = 0.50
corr_value = getCorrelatedfeature(corrmat['Price'], threshold)
corr_value


# In[14]:


corr_value.index.values


# In[15]:


correlated_data = data[corr_value.index]
correlated_data.head()
sns.pairplot(correlated_data)
plt.tight_layout()


# As we can see, RM has positive correlation (linear trend) with Price.
# LSTAT jas negative correlation with Price

# In[16]:


sns.heatmap(correlated_data.corr(), annot= True, annot_kws={'size':12})


# Transform and split data

# In[17]:


x = correlated_data.drop(labels= ['Price'], axis =1)
y = correlated_data['Price']
x.head()


# In[18]:


# Initiate the linear regression model
reg = linear_model.LinearRegression()


# In[19]:


# Split the data into 80% training and 20% testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[20]:


x_train.shape, x_test.shape


# Train the Model

# In[21]:


model = LinearRegression()
model.fit(x_train, y_train)


# In[22]:


y_prediction = model.predict(x_test)


# In[23]:


df= pd.DataFrame(data = [y_prediction, y_test])
df.T


# # Performance of the model 

# In[24]:


from sklearn.metrics import r2_score
score = r2_score (y_test, y_prediction)
mae = mean_absolute_error (y_test, y_prediction)
mse = mean_squared_error(y_test, y_prediction)

print('r2_score:', score)
print ('mae:', mae)
print ('mse:', mse)


# Regression result shows that the features are correlated with the house price

# In[25]:


rows =2
cols =2
fig, ax =plt.subplots(nrows=rows, ncols=cols, figsize =(10,6))

col = correlated_data.columns
index =0

for i in range(rows):
    for j in range(cols):
        sns.regplot(x=correlated_data[col[index]], y=correlated_data['Price'], ax =ax[i][j])
        index = index +1
        
fig.tight_layout()


# Observation:
#     
#     1. RM has positive correlation with price.
#     2. PTRATIO has negative correlation with price.
#     3. LSTAT has negative correlation with price.
#     

# ## Select features which has absolute correlation value more than 60%

# In[26]:


corrmat['Price']


# In[27]:


corr_value = getCorrelatedfeature(corrmat['Price'], 0.6)
corr_value


# In[28]:


correlated_data = data[corr_value.index]
correlated_data.head()


# In[29]:


def get_y_prediction(corrdata):
    x = correlated_data.drop(labels=['Price'], axis =1)
    y = correlated_data['Price']
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.2, random_state= 0)
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_predction = model.predict(x_test)
    return y_predction


# In[30]:


y_prediction = get_y_prediction(correlated_data)


# In[31]:


from sklearn.metrics import r2_score
score = r2_score (y_test, y_prediction)
mae = mean_absolute_error (y_test, y_prediction)
mse = mean_squared_error(y_test, y_prediction)

print('r2_score:', score)
print ('mae:', mae)
print ('mse:', mse)


# # Observation:
# 
# By removing PTRATIO, R2 was increased
# 

# ## Select features which has absolute correlation value more than 70%

# In[32]:


corr_value = getCorrelatedfeature(corrmat['Price'], 0.7)
corr_value


# In[33]:


correlated_data = data[corr_value.index]
correlated_data.head()


# In[34]:


y_prediction = get_y_prediction(correlated_data)
score = r2_score (y_test, y_prediction)
mae = mean_absolute_error (y_test, y_prediction)
mse = mean_squared_error(y_test, y_prediction)

print('r2_score:', score)
print ('mae:', mae)
print ('mse:', mse)

If we remove RM, then R2 score is reduced. Therefore, we will review the performance of model when we only select "RM" feature 
# In[35]:


correlated_data = data[['RM','Price']]
correlated_data.head()


# In[36]:


y_prediction = get_y_prediction(correlated_data)
score = r2_score (y_test, y_prediction)
mae = mean_absolute_error (y_test, y_prediction)
mse = mean_squared_error(y_test, y_prediction)

print('r2_score:', score)
print ('mae:', mae)
print ('mse:', mse)


# Observation:
#     
# The R2 score of model that only use 'RM' as a x variable was reduced compared to that of model that has RM and LSTAT as x variables

# ## Select another combination of features which has absolute correlation value more than 40%

# In[37]:


corr_value = getCorrelatedfeature(corrmat['Price'], 0.4)
corr_value


# In[38]:


correlated_data = data[corr_value.index]
correlated_data.head()


# In[39]:


y_prediction = get_y_prediction(correlated_data)
score = r2_score (y_test, y_prediction)
mae = mean_absolute_error (y_test, y_prediction)
mse = mean_squared_error(y_test, y_prediction)

print('r2_score:', score)
print ('mae:', mae)
print ('mse:', mse)


# Conclusion:
# From building different regression models, and testing the performance, we can conclude that "RM" and "LSTAT" are the features that has the best performance to predict the price
