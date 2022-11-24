# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 00:10:55 2022

@author: nus34
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 18:42:01 2022

@author: nus34
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics

#%%
df = pd.read_csv('india2020.csv')
print(df.head())
#%%
print(df.info())
#%%
print(df.isnull().sum())
#%%
print(df.describe())
#%%

#%%
# df.drop(columns=['col_1', 'col_2','col_N'])

#X = df.drop(columns='PriceOfPaperInDollar', axis=1)
X = df.drop(columns=['State/UT','cost'], axis=1)
y = df['cost']


#%%
df['DoctorsRange'].value_counts().plot(kind='bar')
plt.title('Doctors range in various hospitals of india ')
plt.legend()


