#! Python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Split data
from sklearn.preprocessing import LabelEncoder # Need to encode string data for regression

bank = pd.read_csv("banking.csv")
bank.describe()
for i in bank.columns:
    print(i, " ", bank[i].dtype, " ", bank[i].unique(), '\n')
bank = bank.replace('unknown', np.NaN)
bank = bank.dropna(axis=0)
bank.loc[bank['education'] == 'basic.4y', 'education'] = 'basic'
bank.loc[bank['education'] == 'basic.6y', 'education'] = 'basic'
bank.loc[bank['education'] == 'basic.9y', 'education'] = 'basic'
bank.loc[bank['job'] == 'admin', 'job'] = 'white-collar'
bank.loc[bank['job'] == 'management', 'job'] = 'white-collar'
bank.loc[bank['job'] == 'entrepreneur', 'job'] = 'white-collar'
bank.loc[bank['job'] == 'technician', 'job'] = 'white-collar'
bank.loc[bank['job'] == 'services', 'job'] = 'blue-collar'
bank.loc[bank['job'] == 'housemaid', 'job'] = 'blue-collar'
bank.loc[bank['job'] == 'services', 'job'] = 'blue-collar'
for i in bank.columns:
    print(i, " ", bank[i].dtype)
transformed = bank.select_dtypes(exclude=['number'])
transformed = transformed.apply(LabelEncoder().fit_transform)
transformed = transformed.join(bank.select_dtypes(include=['number']))
for i in transformed.columns:
    if i != 'y':
        if transformed[i].dtype == 'float64':
            transformed[i] = (transformed[i] - transformed[i].mean())/(transformed[i].std())
train, test = train_test_split(transformed, test_size=0.3)
train.to_csv('train_bank.txt', header=None, index=False)
test.to_csv('test_bank.txt', header=None, index=False)
