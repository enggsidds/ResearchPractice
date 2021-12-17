#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


df = pd.read_csv (r'C:\Users\enggs\Documents\RP_CRM_InputData.csv' , encoding = 'unicode_escape')
print (df)


# In[6]:


df.shape


# In[4]:


df.isnull().sum()


# In[5]:


X = df.iloc[:,[3,4]]
X


# In[6]:


X = df.iloc[:,[3,4]].values
X


# In[7]:


plt.scatter(X[...,0],X[...,1])
plt.xlabel('Total Income')
plt.ylabel('Spending Score')
plt.show()


# In[8]:


from sklearn.cluster import KMeans
wcss = []
for i in range (1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans_inertia_)


# In[9]:


from sklearn.cluster import KMeans


# In[10]:


pip install sklearn


# In[11]:


from sklearn.cluster import KMeans
wcss = []
for i in range (1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[12]:


wcss


# In[13]:


plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# In[14]:


kmeans = KMeans(n_clusters=5,init='k-means++',random_state = 0)
Y_kmeans = kmeans.fit_predict(X)
Y_kmeans


# In[15]:


X


# In[16]:


Y_kmeans==0


# In[17]:


X[Y_kmeans==0,0]


# In[18]:


X[Y_kmeans==0,1]


# In[19]:


plt.scatter(X[Y_kmeans==0,0],X[Y_kmeans==0,1])
plt.show()


# In[20]:


plt.scatter(X[Y_kmeans==0,0],X[Y_kmeans==0,1])
plt.scatter(X[Y_kmeans==1,0],X[Y_kmeans==1,1])

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()


# In[21]:


plt.scatter(X[Y_kmeans==0,0],X[Y_kmeans==0,1])
plt.scatter(X[Y_kmeans==1,0],X[Y_kmeans==1,1])
plt.scatter(X[Y_kmeans==2,0],X[Y_kmeans==2,1])

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()


# In[22]:


plt.scatter(X[Y_kmeans==0,0],X[Y_kmeans==0,1])
plt.scatter(X[Y_kmeans==1,0],X[Y_kmeans==1,1])
plt.scatter(X[Y_kmeans==2,0],X[Y_kmeans==2,1])
plt.scatter(X[Y_kmeans==3,0],X[Y_kmeans==3,1])
plt.scatter(X[Y_kmeans==4,0],X[Y_kmeans==4,1])

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()


# In[23]:


plt.scatter(X[Y_kmeans==0,0],X[Y_kmeans==0,1],label='Cluster-1')
plt.scatter(X[Y_kmeans==1,0],X[Y_kmeans==1,1],label='Cluster-2')
plt.scatter(X[Y_kmeans==2,0],X[Y_kmeans==2,1],label='Cluster-3')
plt.scatter(X[Y_kmeans==3,0],X[Y_kmeans==3,1],label='Cluster-4')
plt.scatter(X[Y_kmeans==4,0],X[Y_kmeans==4,1],label='Cluster-5')


# In[25]:


plt.scatter(X[Y_kmeans==0,0],X[Y_kmeans==0,1],label='Cluster-1')
plt.scatter(X[Y_kmeans==1,0],X[Y_kmeans==1,1],label='Cluster-2')
plt.scatter(X[Y_kmeans==2,0],X[Y_kmeans==2,1],label='Cluster-3')
plt.scatter(X[Y_kmeans==3,0],X[Y_kmeans==3,1],label='Cluster-4')
plt.scatter(X[Y_kmeans==4,0],X[Y_kmeans==4,1],label='Cluster-5')

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()


# In[26]:


plt.scatter(X[Y_kmeans==0,0],X[Y_kmeans==0,1],label='Avg Incomoe Avg Shopping')
plt.scatter(X[Y_kmeans==1,0],X[Y_kmeans==1,1],label='Less Income Less Shopping')
plt.scatter(X[Y_kmeans==2,0],X[Y_kmeans==2,1],label='High Income High Shopping')
plt.scatter(X[Y_kmeans==3,0],X[Y_kmeans==3,1],label='High Income Less Shopping')
plt.scatter(X[Y_kmeans==4,0],X[Y_kmeans==4,1],label='Less Income High Shopping')

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()


# In[27]:


plt.scatter(X[Y_kmeans==0,0],X[Y_kmeans==0,1],label='Avg Incomoe Avg Shopping')
plt.scatter(X[Y_kmeans==1,0],X[Y_kmeans==1,1],label='Less Income Less Shopping')
plt.scatter(X[Y_kmeans==2,0],X[Y_kmeans==2,1],label='High Income High Shopping')
plt.scatter(X[Y_kmeans==3,0],X[Y_kmeans==3,1],label='High Income Less Shopping')
plt.scatter(X[Y_kmeans==4,0],X[Y_kmeans==4,1],label='Less Income High Shopping')

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.rcParams['figure.figsize'] = [8, 5]
plt.show()


# In[28]:


plt.scatter(X[Y_kmeans==0,0],X[Y_kmeans==0,1],label='Avg Incomoe Avg Shopping')
plt.scatter(X[Y_kmeans==1,0],X[Y_kmeans==1,1],label='Less Income Less Shopping')
plt.scatter(X[Y_kmeans==2,0],X[Y_kmeans==2,1],label='High Income High Shopping')
plt.scatter(X[Y_kmeans==3,0],X[Y_kmeans==3,1],label='High Income Less Shopping')
plt.scatter(X[Y_kmeans==4,0],X[Y_kmeans==4,1],label='Less Income High Shopping')

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.rcParams['figure.figsize'] = [8, 5]
plt.show()


# In[29]:


df['Target'] = Y_kmeans


# In[30]:


df


# In[31]:


df.to_csv("C:\\Users\\enggs\\Documents\\RP_Results.csv", sep='|')


# In[ ]:




