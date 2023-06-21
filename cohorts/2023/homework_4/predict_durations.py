#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[2]:


import pickle
import pandas as pd


# In[3]:


import numpy as np


# In[4]:


import seaborn as sns


# # Parameters

# In[5]:


year = 2022
month = 2

output_file = './output/duration_predictions.parquet'


# # Load model, data and make predictions

# In[6]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[7]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[8]:


df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')


# In[9]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# # Q1: Standard deviation of predicted durations

# In[10]:


np.std(y_pred)


# In[11]:


sns.displot(y_pred)


# # Q2: Save predictions to file

# In[12]:


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[13]:


df.info()


# In[14]:


df.head()


# In[15]:


df_result = df.loc[:, ["ride_id"]].copy()
df_result["duration_prediction"] = y_pred


# In[16]:


df_result.info()


# In[17]:


df_result.head()


# In[32]:


df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

