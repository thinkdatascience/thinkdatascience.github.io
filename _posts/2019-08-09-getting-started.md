---
title: Multicollinearlity and Variation Inflation Factor
author: Akshay Adlakha & Akshaykumar Rao
date: 2020-12-14 20:55:00 +0800
categories: [Machine Learning, Python]
tags: [VIF]
pin: true
math: true 
mermaid: true
---


## Multicollinearity

Multicollinearity occurs when two or more variables in a dataset are correlated with others. In simpler words, one predictor variable can be used to predict the other. This creates problem when we fit linear regression model and interpret the results. 

Most important assumption of Regression analysis is that each variable should be independent and must contribute to the model. If a multicollinearity or degree of corrleation is high between variables in the data, then the assumption of linear regression gets violated. This needs to be taken care of before fitting a model.

For example, person's height and weight can be the correlated features in dataset.

There are two types of multicollinearity:

- Structural Multicollinearity: This type of multicollinearity is actually not present in the data. Rather it occurs when we create a new feature from other feature. For example, creating the feature x<sup>2</sup> from the feature x. 

- Data Multicollinearity: This type of multicollinearity is already present in the dataset rather than we create a situation of multicollinearity. For example, when we compute correlation coefficients, and we come to have correated features. 

## What Problems Do Multicollinearity Cause?

This can cause two types of problems:

- The model coefficients become so sensitive to small changes in the model. Even a small change can change the value drastically.

- It reduces the statstical power of hypothesis testing. We can't rely the p-values to identify the significant features. Basically, it reduces the precision of the estimated coefficients and becomes difficult to specify the correct model.

So, if there is high of degree of multicollinearity, then there would be high problematic effects. It affects only the correlated variables. Other variables remain unaffected and contribute well to the model.


## Detecting Multicollinearity using Variance Inflation Factor

Now that we know what multicollinearity is, how can we tell if it is present in our data?
Variance Inflation Factors (VIF) is one of the most common techniques used to detect multicollinearity.
VIF measures the severity of multicollinearity in regression analysis.

> VIF provides an index that measures how much the variance of an estimated regression coefficient is increased because of collinearity.

VIF is calculated as follows:

$$ VIF_j  =  {1 \over 1 - R_j^2} $$


Each variable \\(x_j\\) in the dataset is separately treated as the target variable and the remaining variables are treated as the predictors. Next, a linear model is fit 
and \\(R^2\\) value is calculated. Finally, VIF for the target variable is obtained using the above equation.

VIF is always positive and is high when R<sup>2</sup> is closer to 1.  

### Interpreting R<sup>2</sup> and VIF


\\(R_j^2\\) value determines how well an independent variable is described by the other independent variables. 
When \\(R_j^2\\) value is equal to 0, the variance of the remaining independent variables cannot be predicted from the j<sup>th</sup> independent variable. Therefore, when \\(R_j^2\\) = 0 (i.e VIF = 1) which implies that the j<sup>th</sup> variable is not correlated to the remaining variabeles or in other words, multicollinearity does not exist in this regression model. In such case, the variance of \\(_j\\) is not inflated at all. 


> *Note*: As a rule of thumb, a VIF greater than 4 indicates that multicollinearity might exist and further investigation is required. VIF greater than 10 implies a significant multicollinearity that needs to be corrected.







```python
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
```


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




```python
data = data.drop('medv', axis=1)
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
    </tr>
  </tbody>
</table>
</div>




```python
vif = pd.DataFrame()
vif["columns"] = data.columns
vif['VIF'] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
vif
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
      <th>columns</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>crim</td>
      <td>2.100373</td>
    </tr>
    <tr>
      <th>1</th>
      <td>zn</td>
      <td>2.844013</td>
    </tr>
    <tr>
      <th>2</th>
      <td>indus</td>
      <td>14.485758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>chas</td>
      <td>1.152952</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nox</td>
      <td>73.894947</td>
    </tr>
    <tr>
      <th>5</th>
      <td>rm</td>
      <td>77.948283</td>
    </tr>
    <tr>
      <th>6</th>
      <td>age</td>
      <td>21.386850</td>
    </tr>
    <tr>
      <th>7</th>
      <td>dis</td>
      <td>14.699652</td>
    </tr>
    <tr>
      <th>8</th>
      <td>rad</td>
      <td>15.167725</td>
    </tr>
    <tr>
      <th>9</th>
      <td>tax</td>
      <td>61.227274</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ptratio</td>
      <td>85.029547</td>
    </tr>
    <tr>
      <th>11</th>
      <td>b</td>
      <td>20.104943</td>
    </tr>
    <tr>
      <th>12</th>
      <td>lstat</td>
      <td>11.102025</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

