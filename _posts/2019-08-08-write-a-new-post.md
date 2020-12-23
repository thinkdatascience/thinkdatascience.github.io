---
title: Have you ever thought of implementing PCA algorithms from Scratch?
author: Akshay Adlakha & Akshaykumar Rao
date: 2020-12-23 08:10:00 +0800
categories: [Machine Learning, Python]
tags: [Unsupervised Learning]
math: true
mermaid: true
---


## Principal Component Analysis

Principal Component Analysis is a dimensionality reduction technique. It is an unsupervised learning alorithm which reduces the number of features in the dataset while retaining most of the variation or information. In this blog, we will implement this algorithm from scratch in Python.

PCA offers so many benefits other than reducing the dimension of dataset. 

- Reduces Training time: PCA reduces the training time significantly as it results in smaller dataset. Thus, we have less computation time.

- Remove Noise: PCA helps in removing noise from the dataset. It is widely used algorithm to detect outliers.

- Help to visualize data: Mostly, we have high dimensional dataset which is difficult to visualize. It helps in visualize dataset by reducing the number of features.

- Multicollinearity: PCA solves one of the biggest problems - Multicollinearity. It removes highly correlated features by keeping most of information. To read more about this, Check out this [blog](https://thinkdatascience.github.io/posts/PCA/). 

Now, Lets see what are the steps involved in this algorithm:

- Normalize or Scale your data. 

- Obtain the Covariance matrix.

- Compute the eigenvalues and eigenvectors from the Covariance matrix to find the Principal Components.

- Sort the eigenvectors from the highest eigenvalue to the lowest.

- Select the number of principal components.


> Note: PCA is not a feature selection technique. It is rather a feature extraction method as it transforms the data into the principal components.

We know the steps to implement an algorithm and its benefits. Lets see its implementation in Python.

```python
import numpy as np
import pandas as pd

def PCA(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: numpy.ndarray
    """
    X_train -= np.mean(X_train, axis = 0)  
    covarianceM = np.cov(X_train, rowvar = False)
    eigenValues , eigenVectors = np.linalg.eig(covarianceM)
    index = np.argsort(eigenValues)[::-1]
    eigenValues = eigenValues[index]
    eigenVectors = eigenVectors[:,index]

    totalSum =sum(eigenValues)
    variance = [(i/totalSum)*100 for i in sorted(eigenValues, reverse = True)]
    cumValues = np.cumsum(variance)
    
    reducedData = np.dot(X_train, eigenVectors[:,:N])
    return (reducedData)


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
 
x = data.iloc[:,0:4]

target = data.iloc[:,4]

pcadata = PCA(x,2)


```


 
