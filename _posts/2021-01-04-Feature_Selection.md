---
title: Feature Selection Techniques 
author: Akshay Adlakha & Akshaykumar Rao
date: 2021-01-04 14:10:00 +0800
categories: [Machine Learning, Python]
tags: [Machine Learning]
math: true
mermaid: true
---

<b>Feature Selection</b> is the process of choosing only a subset of features(or variables) that are relavant in predicting the target variable. But why is it required ? When you are working with a lot of features, it is common that some of them aren't useful in predictions. Alternately, it is also possible that using a lot of features can lead to overfitting and large training times, making your model produce suboptimal results. Another common problem of working with large number of features is [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality). Moreover, a lot of features will require a lot of training data. By dropping these variables, we are left with a subset of features that can be used to build comparable(or better) modes and even save training time and memory pace.

In this, article we present few effective feature selection techniques.


> Variance Threshold

A feature with a very low variance implies that it is has values close to being constant. Such feature generally have no information that can help add any value to the model and thus, can be dropped from the data set. This can be done using `scikit-learn`'s `VarianceThreshold`.

```python
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
```

We will use the [Santander Customer Satisfaction](https://www.kaggle.com/c/santander-customer-satisfaction/data?select=train.csv) dataset from Kaggle. This dataset has 76k rows and 371 columns! For this article, we'll load only 10k rows as below:


```python
df = pd.read_csv('train.csv', nrows = 10000)
print(df.shape)
df.head()
```

    (10000, 371)
    




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
      <th>ID</th>
      <th>var3</th>
      <th>var15</th>
      <th>imp_ent_var16_ult1</th>
      <th>imp_op_var39_comer_ult1</th>
      <th>imp_op_var39_comer_ult3</th>
      <th>imp_op_var40_comer_ult1</th>
      <th>imp_op_var40_comer_ult3</th>
      <th>imp_op_var40_efect_ult1</th>
      <th>imp_op_var40_efect_ult3</th>
      <th>...</th>
      <th>saldo_medio_var33_hace2</th>
      <th>saldo_medio_var33_hace3</th>
      <th>saldo_medio_var33_ult1</th>
      <th>saldo_medio_var33_ult3</th>
      <th>saldo_medio_var44_hace2</th>
      <th>saldo_medio_var44_hace3</th>
      <th>saldo_medio_var44_ult1</th>
      <th>saldo_medio_var44_ult3</th>
      <th>var38</th>
      <th>TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>23</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>39205.170000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>2</td>
      <td>34</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>49278.030000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>2</td>
      <td>23</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>67333.770000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>2</td>
      <td>37</td>
      <td>0.0</td>
      <td>195.0</td>
      <td>195.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>64007.970000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>2</td>
      <td>39</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117310.979016</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 371 columns</p>
</div>



Next, we'll split the predictors and the target variable as the VarianceThreshold algorithm works only on predictors(X) and not the desired output(Y).


```python
y = df['TARGET']
df = df.drop('TARGET', axis =1)
```

Now we can apply the VarianceThreshold algorithm


```python
vt = VarianceThreshold(threshold=0)
vt.fit(df)
```




    VarianceThreshold(threshold=0)



`threshold` = 0 implies that we want to select those columns which has 0 variance.

The `get.support()` method returns a array with values: False if the feature has 0 variance and True if it doesn't

We can sum over the array to find the count of non-constant features. Similarly, we can count the number of False in the array to find how many features with 0 variance are present in the dataset.


```python
# count of constant features. 
sum(vt.get_support()==False)
```




    85



Therefore we have 85 columns that are constant(having 0 variance)


```python
# column names of constant features:
df.columns[vt.get_support()==False]
```




    Index(['ind_var2_0', 'ind_var2', 'ind_var13_medio_0', 'ind_var13_medio',
           'ind_var18_0', 'ind_var18', 'ind_var27_0', 'ind_var28_0', 'ind_var28',
           'ind_var27', 'ind_var34_0', 'ind_var34', 'ind_var41', 'ind_var46_0',
           'ind_var46', 'num_var13_medio_0', 'num_var13_medio', 'num_var18_0',
           'num_var18', 'num_var27_0', 'num_var28_0', 'num_var28', 'num_var27',
           'num_var34_0', 'num_var34', 'num_var41', 'num_var46_0', 'num_var46',
           'saldo_var13_medio', 'saldo_var18', 'saldo_var28', 'saldo_var27',
           'saldo_var34', 'saldo_var41', 'saldo_var46',
           'delta_imp_amort_var18_1y3', 'delta_imp_amort_var34_1y3',
           'delta_imp_reemb_var17_1y3', 'delta_imp_reemb_var33_1y3',
           'delta_imp_trasp_var17_out_1y3', 'delta_imp_trasp_var33_out_1y3',
           'delta_num_reemb_var17_1y3', 'delta_num_reemb_var33_1y3',
           'delta_num_trasp_var17_out_1y3', 'delta_num_trasp_var33_out_1y3',
           'imp_amort_var18_hace3', 'imp_amort_var18_ult1',
           'imp_amort_var34_hace3', 'imp_amort_var34_ult1', 'imp_var7_emit_ult1',
           'imp_reemb_var13_hace3', 'imp_reemb_var17_hace3',
           'imp_reemb_var17_ult1', 'imp_reemb_var33_hace3', 'imp_reemb_var33_ult1',
           'imp_trasp_var17_in_hace3', 'imp_trasp_var17_out_hace3',
           'imp_trasp_var17_out_ult1', 'imp_trasp_var33_in_hace3',
           'imp_trasp_var33_out_hace3', 'imp_trasp_var33_out_ult1',
           'imp_venta_var44_hace3', 'ind_var7_emit_ult1', 'num_var2_0_ult1',
           'num_var2_ult1', 'num_var7_emit_ult1', 'num_meses_var13_medio_ult3',
           'num_reemb_var13_hace3', 'num_reemb_var17_hace3',
           'num_reemb_var17_ult1', 'num_reemb_var33_hace3', 'num_reemb_var33_ult1',
           'num_trasp_var17_in_hace3', 'num_trasp_var17_out_hace3',
           'num_trasp_var17_out_ult1', 'num_trasp_var33_in_hace3',
           'num_trasp_var33_out_hace3', 'num_trasp_var33_out_ult1',
           'num_venta_var44_hace3', 'saldo_var2_ult1',
           'saldo_medio_var13_medio_hace2', 'saldo_medio_var13_medio_hace3',
           'saldo_medio_var13_medio_ult1', 'saldo_medio_var13_medio_ult3',
           'saldo_medio_var29_hace3'],
          dtype='object')



Let's  drop these constant columns 



```python
df = df.drop(df.columns[vt.get_support()==False], axis = 1)
df.shape
```




    (10000, 285)



After dropping those columns, we are now left with 285 columns.

<b>NOTE: the value of threshold (or Variance) depends on scaling of the data</b>

Alternately, you can also use the fit_transform method to tranform the entire training data and achieve the same results i.e. 285 columns


```python
vt = VarianceThreshold(threshold=0)
df = vt.fit_transform(df)
df.shape
```




    (10000, 285)



> Pearson Correlation

Another common feature selection technique is finding the correlations between the feature in the dataset.

The Pearson correlation coefficient values range between -1 to 1. The values are interpreted as follows:
   * 1: There is a total positive correlation (if one feature goes up, the other also goes up)
   * -1: There is a total negative correlation (if one feature goes down, the other goes up)
   * 0: There is no correlation

If we have a strong correlation between features (/rho > 0.8), then we can keep of the features and drop the others.

<b>NOTE: this statistic measures only the linear correlation between the features. Therefore, if the correlation is 0, it may also mean the there might exist a non-linear relationship between the variables.</b> 



Let's see how correlation can be calculated in python. 

For this purpose, we will use the california housing dataset.


```python
from sklearn.datasets import fetch_california_housing
df =fetch_california_housing()
data = pd.DataFrame(df["data"], columns=df["feature_names"])
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
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.3252</td>
      <td>41.0</td>
      <td>6.984127</td>
      <td>1.023810</td>
      <td>322.0</td>
      <td>2.555556</td>
      <td>37.88</td>
      <td>-122.23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.3014</td>
      <td>21.0</td>
      <td>6.238137</td>
      <td>0.971880</td>
      <td>2401.0</td>
      <td>2.109842</td>
      <td>37.86</td>
      <td>-122.22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.2574</td>
      <td>52.0</td>
      <td>8.288136</td>
      <td>1.073446</td>
      <td>496.0</td>
      <td>2.802260</td>
      <td>37.85</td>
      <td>-122.24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.6431</td>
      <td>52.0</td>
      <td>5.817352</td>
      <td>1.073059</td>
      <td>558.0</td>
      <td>2.547945</td>
      <td>37.85</td>
      <td>-122.25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.8462</td>
      <td>52.0</td>
      <td>6.281853</td>
      <td>1.081081</td>
      <td>565.0</td>
      <td>2.181467</td>
      <td>37.85</td>
      <td>-122.25</td>
    </tr>
  </tbody>
</table>
</div>



We'll now calculate Pearson Correlation and visualize it using Seaborn.


```python
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
correlation_matrix = data.corr()
plt.figure(figsize=(10,10))
hm = sns.heatmap(data.corr(), annot = True) 
plt.show()
```


    
![png](output_21_0.png)
    


As we can see from the chart, there is a strong positive correlation (0.85) between <i>AveBedrms</i> and <i>AveRooms</i>. This is intuitively possible, since the number of bedrooms will increase with increase in number of rooms in a household. 

Also, there is a very strong negative correlation (-0.92) between features <i>Latitude</i> and <i>Longitude</i>.

Now, we can keep one feature and drop the other feature.

<b>NOTE: An important thing to note about correlation is that if there is a strong correlation between your independent features, then you must keep one of them and drop the rest as they generally "duplicate" information.
However, if there is a strong correlation between your independent features and the dependent feature (or the target variable), then you must not drop the correlated feature as they will be an important predictor in your model. </b>

> Mutual Information

This is a univariate feature selection technique to find correlated features. From the sklearn docs:
 
<i>"Mutual information (MI) between two random variables is a non-negative
value, which measures the dependency between the variables. It is equal
to zero if and only if two random variables are independent, and higher
values mean higher dependency.

The function relies on nonparametric methods based on entropy estimation
from k-nearest neighbors distances." </i>

Mutual Info can be calculated using sklearn's `mutual_info_regression` (for regression) and `mutual_info_classif` (for classification).
These 2 methods calculate the MI values between the predictor variables and the target variables. If the MI value for a feature is 0, then you can safely drop the feature, whereas, if the MI value is close to 1, then the feature is a strong predictors of the target variable.

We will continue to use the california housing dataset.


```python
from sklearn.feature_selection import mutual_info_regression
mutual_info = mutual_info_regression(X=data,y=df["target"])
```


```python
mutual_info_df = pd.DataFrame(list(zip(df["feature_names"], mutual_info)))
mutual_info_df.columns = ['Feature_names', 'MI']
mutual_info_df = mutual_info_df.sort_values('MI')
mutual_info_df
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
      <th>Feature_names</th>
      <th>MI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Population</td>
      <td>0.020955</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AveBedrms</td>
      <td>0.024157</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HouseAge</td>
      <td>0.030921</td>
    </tr>
    <tr>
      <th>5</th>
      <td>AveOccup</td>
      <td>0.072708</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AveRooms</td>
      <td>0.103555</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Latitude</td>
      <td>0.371504</td>
    </tr>
    <tr>
      <th>0</th>
      <td>MedInc</td>
      <td>0.387434</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Longitude</td>
      <td>0.402218</td>
    </tr>
  </tbody>
</table>
</div>



As we can see we don't have any feature that is totally independent of the target variable.

> Recursive Feature Elimination (RFE)

RFE is a greedy feature selection approach in which we first build a model with all the features and then recursively remove the least important feature at every iteration. But how does it tell which feature is least important ? For linear models like logistic regression or SVMs, the importance of the feature is decided by the coefficient of each feature whereas, in case of tree-based models like decision trees or random forest, feature importance is decided by the `feature_importances_` method of the model.



Let's see how RFE is implemented in python. We will use the diabetes dataset from sklearn. This is a regresssion task and we will use linear regression as our estimator.


```python
from sklearn.feature_selection import RFE
from sklearn.datasets import load_diabetes
df = load_diabetes() 
X = df["data"]
y = df["target"]
features = df["feature_names"]
data = pd.DataFrame(X, columns=features)
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
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.038076</td>
      <td>0.050680</td>
      <td>0.061696</td>
      <td>0.021872</td>
      <td>-0.044223</td>
      <td>-0.034821</td>
      <td>-0.043401</td>
      <td>-0.002592</td>
      <td>0.019908</td>
      <td>-0.017646</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.001882</td>
      <td>-0.044642</td>
      <td>-0.051474</td>
      <td>-0.026328</td>
      <td>-0.008449</td>
      <td>-0.019163</td>
      <td>0.074412</td>
      <td>-0.039493</td>
      <td>-0.068330</td>
      <td>-0.092204</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.085299</td>
      <td>0.050680</td>
      <td>0.044451</td>
      <td>-0.005671</td>
      <td>-0.045599</td>
      <td>-0.034194</td>
      <td>-0.032356</td>
      <td>-0.002592</td>
      <td>0.002864</td>
      <td>-0.025930</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.089063</td>
      <td>-0.044642</td>
      <td>-0.011595</td>
      <td>-0.036656</td>
      <td>0.012191</td>
      <td>0.024991</td>
      <td>-0.036038</td>
      <td>0.034309</td>
      <td>0.022692</td>
      <td>-0.009362</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.005383</td>
      <td>-0.044642</td>
      <td>-0.036385</td>
      <td>0.021872</td>
      <td>0.003935</td>
      <td>0.015596</td>
      <td>0.008142</td>
      <td>-0.002592</td>
      <td>-0.031991</td>
      <td>-0.046641</td>
    </tr>
  </tbody>
</table>
</div>




```python
#initialize the linear regression model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
```

Now that we have intialized the linear model, we can implement RFE. 
We will use three arguments when we initialize RFE:
* *estimator*: which in this case is the linear model.
* *n_features_to_select*: number of features to select. By default it will select half of the features.
* *step*: number of features to remove at each iteration


```python
#initialie RFE
rfe_init = RFE(estimator= lm,
         n_features_to_select=6,
         step = 1)


# fit RFE
rfe = rfe_init.fit(data, y)
```


```python
# this method returns an array of True/False for if the feature was selected by the model.
print("Selected Features:", rfe.support_)
# this method returns the ranking of the features
print("Feature Rank:", rfe.ranking_)
```

    Selected Features: [False  True  True  True  True  True False False  True False]
    Feature Rank: [5 1 1 1 1 1 3 2 1 4]
    

Let's print which of the 6 features were selected in RFE.


```python
selected_columns = [x for x,y in zip(data.columns,rfe.ranking_) if y == True]
selected_columns
```




    ['sex', 'bmi', 'bp', 's1', 's2', 's5']



Now that we have our subset of immportant features, we can transform our dataset by dropping the rest of the features using the `transform` method.


```python
data_transformed = rfe.transform(data)
data_transformed = pd.DataFrame(data_transformed, columns=selected_columns)
data_transformed.head()
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
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.050680</td>
      <td>0.061696</td>
      <td>0.021872</td>
      <td>-0.044223</td>
      <td>-0.034821</td>
      <td>0.019908</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.044642</td>
      <td>-0.051474</td>
      <td>-0.026328</td>
      <td>-0.008449</td>
      <td>-0.019163</td>
      <td>-0.068330</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.050680</td>
      <td>0.044451</td>
      <td>-0.005671</td>
      <td>-0.045599</td>
      <td>-0.034194</td>
      <td>0.002864</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.044642</td>
      <td>-0.011595</td>
      <td>-0.036656</td>
      <td>0.012191</td>
      <td>0.024991</td>
      <td>0.022692</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.044642</td>
      <td>-0.036385</td>
      <td>0.021872</td>
      <td>0.003935</td>
      <td>0.015596</td>
      <td>-0.031991</td>
    </tr>
  </tbody>
</table>
</div>



> Feature Importances with Tree-based methods

sklearn provides `feature_importances_` method with tree-based models like decision trees, random forests, etc.
Let's see how we can get feature importance from a random forest model.

We'll use the diabetes dataset as above.


```python
# load Diabetes dataset
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
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.038076</td>
      <td>0.050680</td>
      <td>0.061696</td>
      <td>0.021872</td>
      <td>-0.044223</td>
      <td>-0.034821</td>
      <td>-0.043401</td>
      <td>-0.002592</td>
      <td>0.019908</td>
      <td>-0.017646</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.001882</td>
      <td>-0.044642</td>
      <td>-0.051474</td>
      <td>-0.026328</td>
      <td>-0.008449</td>
      <td>-0.019163</td>
      <td>0.074412</td>
      <td>-0.039493</td>
      <td>-0.068330</td>
      <td>-0.092204</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.085299</td>
      <td>0.050680</td>
      <td>0.044451</td>
      <td>-0.005671</td>
      <td>-0.045599</td>
      <td>-0.034194</td>
      <td>-0.032356</td>
      <td>-0.002592</td>
      <td>0.002864</td>
      <td>-0.025930</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.089063</td>
      <td>-0.044642</td>
      <td>-0.011595</td>
      <td>-0.036656</td>
      <td>0.012191</td>
      <td>0.024991</td>
      <td>-0.036038</td>
      <td>0.034309</td>
      <td>0.022692</td>
      <td>-0.009362</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.005383</td>
      <td>-0.044642</td>
      <td>-0.036385</td>
      <td>0.021872</td>
      <td>0.003935</td>
      <td>0.015596</td>
      <td>0.008142</td>
      <td>-0.002592</td>
      <td>-0.031991</td>
      <td>-0.046641</td>
    </tr>
  </tbody>
</table>
</div>



Next, we will fit Random Forest Regressor and plot the feature importances.


```python
# Initialize Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

# fit random forest regressor
rf.fit(data,y)

# Plot feature importances
import matplotlib.pyplot as plt
%matplotlib inline
feature_importances = pd.Series(rf.feature_importances_, 
                                index = data.columns)

feature_importances.plot(kind = 'barh')
plt.show()

```


    
![png](output_45_0.png)
    


The plot above shows the importances of each features. As we can see, features *s5* and *bmi* are the most important features, followed by *bp* and so on.

We can then choose to keep the important features and drop the remaining ones.




