# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 21:14:40 2022

@author: nus34
"""

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
df = pd.read_csv('bedno.csv')
print(df.head())
#%%
print(df.info())
#%%
print(df.isnull().sum())
#%%
print(df.describe())
#%%
df['Noofbed'].value_counts().plot(kind='bar')
plt.ylabel("amount of hospitals that has 31,50 etc no of bed")
plt.xlabel("no of beds for ex: 31,50,100 etc")
plt.legend()
