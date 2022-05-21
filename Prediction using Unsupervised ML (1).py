#!/usr/bin/env python
# coding: utf-8

# <b>THE SPARKS FOUNDATION<b>
#     
# <b>TASK 2: Prediction using Unsupervised ML<b>
#     
# <b>Objective: To predict the optimum number of clusters and represent it visually.<b>
#     
# <b>Dataset: https://bit.ly/3kXTdox <b>
#     
# <b>Author: TANUSHREE KHAITAN<b>

# In[15]:


# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets


# In[16]:


# Reading the Iris dataset 

iris = pd.read_csv(r"C:\Users\Lenovo\Downloads\Iris.csv")

iris_df = pd.DataFrame(iris, columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
iris_df


# In[4]:


# Finding the optimum number of clusters for k-means classification

x = iris_df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Plotting the results onto a line graph

plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[8]:


# Applying kmeans to the dataset

kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[13]:


# Visualising the clusters

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'coral', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'cadetblue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'darkkhaki', label = 'Iris-virginica')

# Plotting the centroids of the clusters

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'crimson', label = 'Centroids')

plt.legend()

