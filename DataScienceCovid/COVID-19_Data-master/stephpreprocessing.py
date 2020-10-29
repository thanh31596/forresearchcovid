#!/usr/bin/env python
# coding: utf-8

# In[8]:


import tensorflow


# In[11]:


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pydot
from io import StringIO
from sklearn.tree import export_graphviz
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from pandas.plotting import register_matplotlib_converters

# In[12]:


from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# In[13]:


import keras


# In[14]:


def convert2matrix(data_arr, look_back):
 X, Y =[], []
 for i in range(len(data_arr)-look_back):
  d=i+look_back  
  X.append(data_arr[i:d,0])
  Y.append(data_arr[d,0])
 return np.array(X), np.array(Y)


# In[15]:


def model_dnn(look_back):
    model=Sequential()
    model.add(Dense(units=32,input_dim=look_back, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',  optimizer='adam',metrics = ['mse', 'mae'])
    return model


# In[16]:


def model_loss(history):
   plt.figure(figsize=(8,4))
   plt.plot(history.history['loss'], label='Train Loss')
   plt.plot(history.history['val_loss'], label='Test Loss')
   plt.title('model loss')
   plt.ylabel('loss')
   plt.xlabel('epochs')
   plt.legend(loc='upper right')
   plt.show();


# In[17]:


def prediction_plot(testY, test_predict):
   len_prediction=[x for x in range(len(testY))]
   plt.figure(figsize=(8,4))
   plt.plot(len_prediction, testY[:1], marker='.', label="actual")
   plt.plot(len_prediction, test_predict[:1], 'r', label="prediction")
   plt.tight_layout()
   sns.despine(top=True)
   plt.subplots_adjust(left=0.07)
   plt.ylabel('Ads Daily Spend', size=15)
   plt.xlabel('Time step', size=15)
   plt.legend(fontsize=15)
   plt.show();


# In[18]:


from keras.callbacks import EarlyStopping
def model_rnn(look_back):
  model=Sequential()
  model.add(SimpleRNN(units=32, input_shape=(1,look_back), activation="relu"))
  model.add(Dense(8, activation='relu'))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error',  optimizer='adam',metrics = ['mse', 'mae'])
  return model


# In[19]:


def prediction_plot2(testY, test_predict,look_back):
    len_prediction=[x for x in range(len(testY)-look_back)]
    plt.plot(len_prediction, testY[:1], marker='.', label="actual")
    plt.plot(len_prediction, test_predict[:1], 'r', label="prediction")
    plt.tight_layout()
    sns.despine(top=True)
    plt.subplots_adjust(left=0.07)
    plt.ylabel('Ads Daily Spend', size=15)
    plt.xlabel('Time step', size=15)
    plt.legend(fontsize=15)
    plt.show()


# In[ ]:




