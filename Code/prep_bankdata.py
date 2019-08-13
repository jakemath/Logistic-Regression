
# coding: utf-8

# In[91]:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Split data
from sklearn.preprocessing import LabelEncoder # Need to encode string data for regression


# In[92]:

bank = pd.read_csv("banking.csv")
bank.describe()

# In[93]:

# Print unique values in each column.
# Notice unknown values exist in many columns

for i in bank.columns:
    print(i, " ", bank[i].dtype, " ", bank[i].unique(), '\n')

# In[94]:

# Replace all unknown values with NaN
bank = bank.replace('unknown', np.NaN)
bank = bank.dropna(axis=0)

# In[95]:

# Simplify dataset

bank.loc[bank['education'] == 'basic.4y', 'education'] = 'basic' # Group basic educations together
bank.loc[bank['education'] == 'basic.6y', 'education'] = 'basic'
bank.loc[bank['education'] == 'basic.9y', 'education'] = 'basic'

bank.loc[bank['job'] == 'admin', 'job'] = 'white-collar' # Group white-collar jobs together
bank.loc[bank['job'] == 'management', 'job'] = 'white-collar'
bank.loc[bank['job'] == 'entrepreneur', 'job'] = 'white-collar'
bank.loc[bank['job'] == 'technician', 'job'] = 'white-collar'

bank.loc[bank['job'] == 'services', 'job'] = 'blue-collar' # Group blue-collar/service jobs together
bank.loc[bank['job'] == 'housemaid', 'job'] = 'blue-collar'
bank.loc[bank['job'] == 'services', 'job'] = 'blue-collar'

# In[96]:

for i in bank.columns:
    print(i, " " , bank[i].dtype)

# In[98]:

# Need to transform string data to numeric values before applying model

transformed = bank.select_dtypes(exclude = ['number']) # Select all non-numeric data columns
transformed = transformed.apply(LabelEncoder().fit_transform) # Transform string values into numeric values
transformed = transformed.join(bank.select_dtypes(include = ['number'])) # Join the newly encoded columns to the rest of the frame

for i in transformed.columns:
    if i != 'y':
        if transformed[i].dtype == 'float64':
            transformed[i] = (transformed[i] - transformed[i].mean())/(transformed[i].std())

# In[99]:

train, test = train_test_split(transformed, test_size = 0.3) # Split data, 30% for testing

print(train.to_csv(header=None, index=None), file=open("train_bank.txt", "a"))
print(test.to_csv(header=None, index=None), file=open("test_bank.txt", "a"))

