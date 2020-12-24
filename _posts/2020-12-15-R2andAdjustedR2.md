---
title: R<sup>2</sup> and Adjusted R<sup>2</sup> 
author: Akshay Adlakha & Akshaykumar Rao
date: 2020-12-15 08:10:00 +0800
categories: [Machine Learning, Python]
tags: [Evaluation Metrics, Linear Model]
math: true
mermaid: true
---


## R<sup>2</sup> and Adjusted R<sup>2</sup> 

After fitting a regression model, you have to determine how well the model fits to the data. Is it good enough to generalize to unseen data? or Does it do a good job of explaining changes in the dependent variable?. In this post, we explore R-Squared and Adjusted R-Squared, its sigficance and limitations. 

![upload-image](/assets/img/sample/r2.jpeg)

> R<sup>2</sup> is a goodness of fit measure. It tells us that how close the data are to the fitted line. 

R<sup>2</sup> is calculated as follows:

R<sup>2</sup> = Explained variation / Total variation

or, R<sup>2</sup> = 1 −  Unexplained Variation / Total Variation


This gives us the percentage of the response variable variation that is explained by our linear regression model.

It is also known by other names such as Coefficient of Determination, or the coefficient of multiple determination for multiple regression.

We can visualize the effect of R-Squared value. It represents how data points are scattered around the regression line.

![upload-image](/assets/img/sample/Rsquared.png)

Let's say, R-Squared value for the model on the left is between 15-20% and for the model on the right is 80%. This means that on the left 15%-20% of the variance explained by the model and the model explains 80% of the variance on the right. In general, We want our data points close to the regression line. However, in practice we don't get a model with 100% R-Squared value i.e all the data points fall exactly on the regression line.

## Limitations of R-Squared

R-Squared value gives an estimate of the relationship between input features and the response feature. However it has some limitations.

- It doesn't inform whether a chosen model is good or bad.
- Never tells that data and predictions are biased or not.

A model can have a low or a high R-Squared value. And, it is not necessarily good or bad. A model with low R-Squared value can be a good model or a model with high R-Squared value can be a poor fitted model, and vice-versa. It is recommended to see residual plot or Adjusted R-Squared to interpret the linear model. 

## Adjusted R-Squared

> The adjusted R-squared lets us to compare the descriptive power of regression models that include multiple predictors in the model. 

It solves the problem associated with R-Squared. Whenever we add a new predictor to a model, R-Squared value always increase irrespective of the fact that whether the added predictor is significant or not. So, it looks better for the fact that it has more variable but in reality it is not. It is necessary that it's value to be adjusted. Therefore, Adjusted R-Squared came to be better metric to evaluate a model. It only increases whenever a significant predictor is added to a model or, a given predictor improves the model more than what is predicted by chance otherwise it decreases. Whenever we have an overfitting situation, a high value of R-squared is obtained even though the model has less generalizability. 

## Python Implementation

Let's see an example of calculating R-Squared and Adjusted R-Squared.

In python, `r2_score` metric is available in `sklearn.metrics` to calculate R-Squared value.

We start by importing the necesary packages:

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
```

For this article, we chose the Boston House Prices data set, which is also available in sklearn package. More info [here]{https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html}. This dataset has 503 rows and 13 columns.

```python
data  = pd.read_csv("boston.csv")
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>crim</th>
      <th>zn</th>
      <th>indus</th>
      <th>chas</th>
      <th>nox</th>
      <th>rm</th>
      <th>age</th>
      <th>dis</th>
      <th>rad</th>
      <th>tax</th>
      <th>ptratio</th>
      <th>b</th>
      <th>lstat</th>
      <th>medv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1</td>
      <td>296</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>



We then drop few variables such as `ptratio`, `rm`, `nox`, `tax`, `b`.
```python
df = data.drop(['ptratio', 'rm', 'nox', 'tax', 'b'], axis=1)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>crim</th>
      <th>zn</th>
      <th>indus</th>
      <th>chas</th>
      <th>age</th>
      <th>dis</th>
      <th>rad</th>
      <th>lstat</th>
      <th>medv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>



Then, We normalize the data.
```python
def normalize_data(data, columns):
    data_columns = data.columns
    scaler = MinMaxScaler()
    for col in columns:
        data[col] = scaler.fit_transform(data[[col]])
    return pd.DataFrame(data, columns=data_columns)
```


```python
df_normalized = normalize_data(df,[ 'zn', 'indus', 'chas', 'age', 'dis', 'rad', 'lstat', 'medv'] )
df_normalized
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>crim</th>
      <th>zn</th>
      <th>indus</th>
      <th>chas</th>
      <th>age</th>
      <th>dis</th>
      <th>rad</th>
      <th>lstat</th>
      <th>medv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>0.18</td>
      <td>0.067815</td>
      <td>0.0</td>
      <td>0.641607</td>
      <td>0.269203</td>
      <td>0.000000</td>
      <td>0.089680</td>
      <td>0.422222</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.00</td>
      <td>0.242302</td>
      <td>0.0</td>
      <td>0.782698</td>
      <td>0.348962</td>
      <td>0.043478</td>
      <td>0.204470</td>
      <td>0.368889</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.00</td>
      <td>0.242302</td>
      <td>0.0</td>
      <td>0.599382</td>
      <td>0.348962</td>
      <td>0.043478</td>
      <td>0.063466</td>
      <td>0.660000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.00</td>
      <td>0.063050</td>
      <td>0.0</td>
      <td>0.441813</td>
      <td>0.448545</td>
      <td>0.086957</td>
      <td>0.033389</td>
      <td>0.631111</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.00</td>
      <td>0.063050</td>
      <td>0.0</td>
      <td>0.528321</td>
      <td>0.448545</td>
      <td>0.086957</td>
      <td>0.099338</td>
      <td>0.693333</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>501</th>
      <td>0.06263</td>
      <td>0.00</td>
      <td>0.420455</td>
      <td>0.0</td>
      <td>0.681771</td>
      <td>0.122671</td>
      <td>0.000000</td>
      <td>0.219095</td>
      <td>0.386667</td>
    </tr>
    <tr>
      <th>502</th>
      <td>0.04527</td>
      <td>0.00</td>
      <td>0.420455</td>
      <td>0.0</td>
      <td>0.760041</td>
      <td>0.105293</td>
      <td>0.000000</td>
      <td>0.202815</td>
      <td>0.346667</td>
    </tr>
    <tr>
      <th>503</th>
      <td>0.06076</td>
      <td>0.00</td>
      <td>0.420455</td>
      <td>0.0</td>
      <td>0.907312</td>
      <td>0.094381</td>
      <td>0.000000</td>
      <td>0.107892</td>
      <td>0.420000</td>
    </tr>
    <tr>
      <th>504</th>
      <td>0.10959</td>
      <td>0.00</td>
      <td>0.420455</td>
      <td>0.0</td>
      <td>0.889804</td>
      <td>0.114514</td>
      <td>0.000000</td>
      <td>0.131071</td>
      <td>0.377778</td>
    </tr>
    <tr>
      <th>505</th>
      <td>0.04741</td>
      <td>0.00</td>
      <td>0.420455</td>
      <td>0.0</td>
      <td>0.802266</td>
      <td>0.125072</td>
      <td>0.000000</td>
      <td>0.169702</td>
      <td>0.153333</td>
    </tr>
  </tbody>
</table>
<p>506 rows × 9 columns</p>
</div>


Then, we fit a linear regression model on the dataset.

```python
def linear_model_predictions(data):
    lm = LinearRegression()
    lm.fit(data.iloc[:,:-1], data.iloc[:,-1])
    return lm.predict(data.iloc[:,:-1])
```
Adjusted R-Squared is not directly available in the package. We defined a custom function to calculate it.

```python
def calculate_adj_r2(r2, n, p):    #adjusted r2
    return(1-(1-r2)*(n-1)/(n-p-1))
```
Now, we are calculating the R-Squared and Adjusted R-Squared value.

```python
y_pred = linear_model_predictions(df_normalized)
r2 = r2_score(df_normalized.iloc[:,-1], y_pred) # R-squared
adj_r2 = calculate_adj_r2(r2, df_normalized.shape[0], df_normalized.shape[1])

print("R-squared of the model:", r2)
print("Adjusted R-squared of the model:", adj_r2)

```

    R-squared of the model: 0.6353055729967301
    Adjusted R-squared of the model: 0.628688133797074
    




