#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
#importing tools which we are going to use them.


# In[2]:


df = pd.read_csv('C:/Users/Mert/Downloads/cs_go_data.csv', encoding='latin-1')
#reading our data.


# In[3]:


df.head() #observe a few values


# In[4]:


(df["ct_score"] > 16).sum()


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.info()


# In[8]:


df.count()


# In[9]:


df.isnull().sum()


# In[10]:


sn.histplot(data=df, x = 'round_winner') #drop a histogram of round winner


# In[11]:


sn.set(rc={'figure.figsize':(10,8)})


# In[12]:


sn.boxplot(y='time_left', data=df)


# In[13]:


def moneywin(df):
    count=0
    for case in range(df.shape[0]):
        if df.loc[case][9]>=df.loc[case][10]:
            if df.loc[case][-1]=='CT':
                count+=1
        else:
            if df.loc[case][-1]=='T':
                count+=1
            
    return count


# In[15]:


df.loc[0]


# In[16]:


(df['ct_money'] >=df['t_money']).value_counts()


# In[18]:


df['round_winner'].value_counts()


# In[19]:


(df['ct_money'] < df['t_money']).value_counts()


# In[20]:


(df['ct_money'] >= df['t_money'])&(df['round_winner']=='CT')


# In[21]:


(df['ct_money'] >= df['t_money'])&(df['round_winner']=='T')


# In[22]:


(df['ct_money'] >= df['t_money'])&(df['round_winner']=='T').value_counts()


# In[24]:


(df['ct_money'] >= df['t_money'])&(df['round_winner']=='CT').value_counts()


# In[25]:


((df['ct_money']>=df['t_money']) & (df['round_winner']=='CT')).value_counts()


# In[26]:


((df['ct_money']>=df['t_money']) & (df['round_winner']=='T')).value_counts()


# In[27]:


sn.boxplot(y='ct_score', data=df)


# In[28]:


sn.histplot(x='map', data=df)


# In[29]:


df_2 = df[['map','bomb_planted','round_winner']].copy() # we have 3 columns that have data that we can't use, so we need to modify this data


# In[30]:


df_2.head()


# In[31]:


df_2.info()


# In[33]:


df_2['round_winner'] = df_2['round_winner'].map({'CT':0, 'T':1})


# In[34]:


df_2.head()


# In[35]:


df_2.info()


# In[36]:


df_2["bomb_planted"] = df_2["bomb_planted"].astype(int)


# In[37]:


df_2.head()


# In[38]:


df_2=pd.get_dummies(df_2, columns=['map'])


# In[39]:


df_2.head()


# In[40]:


df['round_winner'] = df['round_winner'].map({'CT':0,'T':1})
df["bomb_planted"] = df["bomb_planted"].astype(int)
df=pd.get_dummies(df, columns=['map'])


# In[41]:


df.head()


# In[42]:


sn.heatmap(df.corr())


# In[44]:


from sklearn.model_selection import *
from sklearn.metrics import *


from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


# In[45]:


y = df["round_winner"]
X = df.drop(["round_winner"],axis=1)


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[47]:


models=[]
models.append(LogisticRegression(solver='liblinear'))
models.append(DecisionTreeClassifier())
models.append(RandomForestClassifier())


# In[48]:


accuracy_models=[]
cross_val_scores=[]
for model in models:
    model.fit(X_train,y_train)
    y_hat=model.predict(X_test)
    accuracy_models.append(accuracy_score(y_test,y_hat))
    cross_val_scores.append(cross_val_score(model,X,y,cv=5,scoring='accuracy', n_jobs=-1).mean())


# In[50]:


accuracy_models


# In[51]:


cross_val_scores


# In[52]:


results = {'Models': models,
          'Accuracy Score': accuracy_models,
          'Cross Validation Score': cross_val_scores}

results=pd.DataFrame(results)


# In[53]:


results


# In[54]:


importances = models[2].feature_importances_
indices = np.argsort(importances)


# In[55]:


importances


# In[56]:


indices


# In[57]:


X_train.columns[indices]


# In[58]:


print(X_train.columns[indices][-1])
print(X_train.columns[indices][-2])
print(X_train.columns[indices][-3])
print(X_train.columns[indices][-4])
print(X_train.columns[indices][-5])


# In[59]:


plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), X_train.columns[indices])
plt.title('Importance of the Columns')


# In[62]:





# In[ ]:




