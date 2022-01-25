#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error , mean_absolute_percentage_error , mean_squared_error
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv('Admission_Prediction.csv')
print(df.head())


# In[3]:


df.drop("Serial No.",axis=1,inplace=True)


# In[4]:


print(df.isna().sum())


# In[5]:


print(df.info())


# In[6]:


print(df.describe())


# In[7]:


print(df.columns)


# In[8]:


for i in df.columns[:-1]:
    sns.regplot(x=i,y='Chance of Admit',data=df)
    plt.show()


# ###### Here We can see tht Every Feature has linear relationship with the target column - Cahnce of Admission

# In[9]:


print('percentage Missing Values',df.isna().sum()[df.isna().sum()!=0].sort_values()/df.shape[0]*100 , sep='\n' )

sns.barplot(df.isna().sum()[df.isna().sum()!=0].sort_values().index , df.isna().sum()[df.isna().sum()!=0].sort_values())
plt.show()

# In[10]:


X = df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[12]:


imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# In[13]:


X_train , X_test =pd.DataFrame(X_train) ,  pd.DataFrame(X_test)
print(X_train.isna().sum() , X_test.isna().sum())


# In[14]:


plt.figure(figsize=(8,24))
n=1
for i in df.columns:
    plt.subplot(8,1,n)
    sns.histplot(x=df[i])
    n+=1
plt.tight_layout()
plt.show()


# In[15]:


scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)              


# In[16]:


print(X_train , X_test)


# In[17]:



svm_model = SVR()
grid=  {'kernel' : ['linear', 'poly', 'rbf'] , 'degree':[3,4,5] }
grid_model = GridSearchCV(svm_model , grid)
grid_model.fit(X_train , y_train)
pred= grid_model.predict(X_test)


# In[18]:


print('SVM  :--  ')
print('mean absoulte error :',mean_absolute_error(y_test , pred) )
print('mean absoulte percentage error :',mean_absolute_percentage_error(y_test , pred) )
print('mean squared error :',mean_squared_error(y_test , pred) )
print('root mean absoulte error :',np.sqrt(mean_absolute_error(y_test , pred)) )


# In[19]:


rf = RandomForestRegressor()
grid = {'n_estimators':[100,200] , 'max_depth' :[10,20,30,100] }
rf_grid = GridSearchCV(rf , grid)
rf_grid.fit(X_train , y_train)
pred = rf_grid.predict(X_test)


# In[20]:


print('Random Forest Regressor  :--  ')
print('mean absoulte error :',mean_absolute_error(y_test , pred) )
print('mean absoulte percentage error :',mean_absolute_percentage_error(y_test , pred) )
print('mean squared error :',mean_squared_error(y_test , pred) )
print('root mean absoulte error :',np.sqrt(mean_absolute_error(y_test , pred)) )


# In[21]:


import joblib
joblib.dump(rf_grid , 'Final_Model.pickle')
joblib.dump(scaler ,'scaler_object.pickle')



