#!/usr/bin/env python
# coding: utf-8

# # Churn Prediction Using ANN

# In[226]:


import numpy as np
import pandas as pd
import matplotlib 
from matplotlib import pyplot as plt
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Reading the file


# df=pd.read_csv("C:/project/churn/Telecom.csv")
# df.sample(5)

# In[ ]:


#Checking the type of variables


# In[228]:


df.drop("customerID",axis="columns",inplace=True)
df.dtypes


# In[229]:


df.TotalCharges.values


# In[230]:


df.MonthlyCharges.values


# In[ ]:


# Converting object type variable to numeric


# In[231]:


pd.to_numeric(df.TotalCharges,errors='coerce').isnull()


# In[ ]:


#Identifying the number of blank 


# In[232]:


df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()].shape


# In[ ]:


#Locating the type of nul value


# In[233]:


df.iloc[488]["TotalCharges"]


# In[ ]:


#Removing the blank data cells


# In[234]:


df1=df[df.TotalCharges!=' ']
df1.shape


# In[268]:


df1.TotalCharges=pd.to_numeric(df1.TotalCharges)


# In[267]:


df1.dtypes


# In[ ]:


#Data Visualization


# In[237]:


import matplotlib.pyplot as plt
from importlib import reload
plt=reload(plt)
tenure_churn_no=df1[df1.Churn=='No'].tenure
tenure_churn_yes=df1[df1.Churn=='Yes'].tenure
plt.xlabel('Tenure')
plt.ylabel("Number of Customers")
plt.title("Customer Churn Prediction Visualization")

plt.hist([tenure_churn_yes,tenure_churn_no],color=["green","orange"],label=["Churn_yes","Churn_no"])
plt.legend()


# In[238]:


import matplotlib.pyplot as plt
from importlib import reload
plt=reload(plt)
tenure_churn_no=df1[df1.Churn=='No'].MonthlyCharges
tenure_churn_yes=df1[df1.Churn=='Yes'].MonthlyCharges
plt.xlabel('MonthlyCharges')
plt.ylabel("Number of Customers")
plt.title("Customer Churn Prediction Visualization")

plt.hist([tenure_churn_yes,tenure_churn_no],color=["green","orange"],label=["Churn_yes","Churn_no"])
plt.legend()


# In[239]:


for column in df:
    if df[column].dtypes=="object":
        print( f'{column}:{df[column].unique()}')
    


# In[271]:


df1.replace("No internet service","No",inplace=True)
df1.replace("No phone service","No",inplace=True)


# In[ ]:


#converting yes and no to 1 and 0


# In[270]:


df1.replace("No","0",inplace=True)
df1.replace("Yes","1",inplace=True)
df1.replace("Female","1",inplace=True)
df1.replace("Male","0",inplace=True)


# In[242]:


for column in df1:
    if df1[column].dtypes=="object":
        print( f'{column}:{df1[column].unique()}')


# In[ ]:


#one hot encoding for categorical columns


# In[243]:


df2 = pd.get_dummies(data=df1, columns=['InternetService','Contract','PaymentMethod'])
df2.columns


# df2

# In[ ]:


#FEATURE SCALING


# In[244]:


cols_to_scale=["tenure","MonthlyCharges","TotalCharges"]
import sklearn
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df2[cols_to_scale]=scaler.fit_transform(df2[cols_to_scale])


# In[245]:


df2.sample(4)


# In[ ]:


#TRAIN TEST SPLIT


# In[246]:


X = df2.drop('Churn',axis='columns')
y = df2['Churn']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)
X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)


# In[247]:


X_train.shape


# In[ ]:


#BUILDING MODEL USING ANN(TENSOR FLOW AND KERAS)


# In[248]:


import tensorflow as tf
from tensorflow import keras


model = keras.Sequential([
    keras.layers.Dense(20, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)


# In[249]:


X_test= np.asarray(X_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)
model.evaluate(X_test, y_test)


# In[250]:


yp = model.predict(X_test)
yp[:5]


# In[251]:


y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)


# In[ ]:


#CLASSIFICATION_REPORT


# In[252]:


from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test,y_pred))


# In[ ]:


#CONFUSION MATRIX


# In[253]:


import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[254]:


y_test.shape


# In[ ]:


#ACCURACY


# In[261]:


round((875+230)/(875+230+178+124),2)


# In[ ]:


#Precision for 0 class. 


# In[262]:


round(875/(875+178),2)


# In[ ]:


#Precision for 1 class


# In[263]:


round(230/(230+124),2)


# In[ ]:


#Recall for 0 class


# In[264]:


round(875/(875+124),2)


# In[ ]:


Recall for 1 class


# In[265]:


round(230/(230+178),2)

