---
title: Churn Prediction
author: Akshay Adlakha & Akshaykumar Rao
date: 2020-12-15 08:10:00 +0800
categories: [Blogging, Tutorial]
tags: [writing]
math: true
mermaid: true
---


## Churn Prediction

![png](/assets/img/sample/churn.png)

Customer churn, also known as customer attrition. It happens when customers stop doing business with a company for some reasons. So, The companies are now interested in finding these customers and look for the reasons why they are leaving. This could help in providing better services to the customera. Moreover, they know that the process of finding a new customer is more time consuming than retaining the one one.  

In this post, we will create a customer churn prediction model using Customer Churn dataset. We will implement an Artificial Neural Network (Tensorflow) to model churned customers, numpy and pandas for data crunching, and matplotlib for visualizations. Moreover, We will try different sampling methods such as Undersampling, Simple Oversampling, SMOTE (Synthetic Minority Oversampling Technique) and Focal Loss to overcome an issue of Imbalanced classes and compare the results. 

The code can be used with another dataset with a few minor adjustments to train the baseline model. 

You can get the dataset from [here.](https://www.kaggle.com/blastchar/telco-customer-churn)

## Libraries


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
from tensorflow import keras
```

## Dataset


```python
data = pd.read_csv('customer_churn.csv')
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe{
    display: block;
    overflow-x: auto;
  }
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
      <th>customerID</th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>...</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>1</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-GNVDE</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>34</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>1889.5</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-QPYBK</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-CFOCW</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>45</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
data.shape
```




    (7043, 21)



In this dataset, we have 7043 samples and 21 features columns. Following are the types of each columns.


```python
data.dtypes
```




    customerID           object
    gender               object
    SeniorCitizen         int64
    Partner              object
    Dependents           object
    tenure                int64
    PhoneService         object
    MultipleLines        object
    InternetService      object
    OnlineSecurity       object
    OnlineBackup         object
    DeviceProtection     object
    TechSupport          object
    StreamingTV          object
    StreamingMovies      object
    Contract             object
    PaperlessBilling     object
    PaymentMethod        object
    MonthlyCharges      float64
    TotalCharges         object
    Churn                object
    dtype: object




```python
data['Churn'].value_counts()
```




    No     5174
    Yes    1869
    Name: Churn, dtype: int64



We have Customer ID in this dataset which would be of no use for modelling. So, We can drop this column. Moreover, we see TotalCharges column to be object type but it should be float type. So, We will now start exploring our data and do some feature engineering.


```python
data.drop('customerID',axis='columns',inplace=True)
```


```python
data['TotalCharges'].values
```




    array(['29.85', '1889.5', '108.15', ..., '346.45', '306.6', '6844.5'],
          dtype=object)




```python
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'],errors='coerce')
```


```python
data.isna().sum()
```




    gender               0
    SeniorCitizen        0
    Partner              0
    Dependents           0
    tenure               0
    PhoneService         0
    MultipleLines        0
    InternetService      0
    OnlineSecurity       0
    OnlineBackup         0
    DeviceProtection     0
    TechSupport          0
    StreamingTV          0
    StreamingMovies      0
    Contract             0
    PaperlessBilling     0
    PaymentMethod        0
    MonthlyCharges       0
    TotalCharges        11
    Churn                0
    dtype: int64



We have got some missing values in TotalCharges column. These are just 11 samples so, we can ignore them. Otherwise, We can use imputation methods to impute values.


```python
data.dropna(subset=['TotalCharges'], inplace=True)
```


```python
data.shape
```




    (7032, 20)



Our target column has two values Yes or No. Whether a customer will an organization or not. We will see our tenure columns and interpret how long or an average customer is staying.


```python
data_tenure_yes = data[data['Churn'] =='Yes'].tenure
data_tenure_no = data[data['Churn'] == 'No'].tenure
data_tenure_yes,data_tenure_no
```




    (2        2
     4        2
     5        8
     8       28
     13      49
             ..
     7021    12
     7026     9
     7032     1
     7034    67
     7041     4
     Name: tenure, Length: 1869, dtype: int64, 0        1
     1       34
     3       45
     6       22
     7       10
             ..
     7037    72
     7038    24
     7039    72
     7040    11
     7042    66
     Name: tenure, Length: 5163, dtype: int64)




```python
plotdata = [data_tenure_yes,data_tenure_no]
plt.hist(plotdata,label = ['Yes','No'])
plt.legend()
plt.show()
```


![png](/assets/img/sample/output_19_0.png)


From this histogram, we can see that if a customer is staying for a long period of time. He is continuing to stay with them. There are a lot of people who left in the first year of joining.


```python
data_monthly_yes = data[data['Churn'] =='Yes']['MonthlyCharges']
data_monthly_no = data[data['Churn'] == 'No']['MonthlyCharges']
data_monthly_yes,data_monthly_no
```




    (2        53.85
     4        70.70
     5        99.65
     8       104.80
     13      103.70
              ...  
     7021     59.80
     7026     44.20
     7032     75.75
     7034    102.95
     7041     74.40
     Name: MonthlyCharges, Length: 1869, dtype: float64, 0        29.85
     1        56.95
     3        42.30
     6        89.10
     7        29.75
              ...  
     7037     21.15
     7038     84.80
     7039    103.20
     7040     29.60
     7042    105.65
     Name: MonthlyCharges, Length: 5163, dtype: float64)




```python
plotdatamonthly = [data_monthly_yes,data_monthly_no]
plt.hist(plotdatamonthly,label = ['Yes','No'])
plt.legend()
plt.show()
```


![png](/assets/img/sample/output_22_0.png)



```python
data_total_yes = data[data['Churn'] =='Yes']['TotalCharges']
data_total_no = data[data['Churn'] == 'No']['TotalCharges']
data_total_yes,data_total_no
```




    (2        108.15
     4        151.65
     5        820.50
     8       3046.05
     13      5036.30
              ...   
     7021     727.80
     7026     403.35
     7032      75.75
     7034    6886.25
     7041     306.60
     Name: TotalCharges, Length: 1869, dtype: float64, 0         29.85
     1       1889.50
     3       1840.75
     6       1949.40
     7        301.90
              ...   
     7037    1419.40
     7038    1990.50
     7039    7362.90
     7040     346.45
     7042    6844.50
     Name: TotalCharges, Length: 5163, dtype: float64)




```python
plotdatatotal = [data_total_yes,data_total_no]
plt.hist(plotdatatotal,label = ['Yes','No'])
plt.legend()
plt.show()
```


![png](/assets/img/sample/output_24_0.png)



```python
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
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>1</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>34</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>1889.50</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>45</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>



We have some columns with values 'No Phone service' or 'No Internet service'. It actually means 'No'. So, we can replace these values with No. And, then can perform One Hot encoding.


```python
data.replace('No phone service','No',inplace=True)
```


```python
data.replace('No internet service','No',inplace=True)
```


```python
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
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>1</td>
      <td>No</td>
      <td>No</td>
      <td>DSL</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>34</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>1889.50</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>45</td>
      <td>No</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.preprocessing import LabelEncoder
```


```python
lb= LabelEncoder()
```


```python
columns = []
for i in data.columns:
    print(i,data[i].unique())
```

    gender ['Female' 'Male']
    SeniorCitizen [0 1]
    Partner ['Yes' 'No']
    Dependents ['No' 'Yes']
    tenure [ 1 34  2 45  8 22 10 28 62 13 16 58 49 25 69 52 71 21 12 30 47 72 17 27
      5 46 11 70 63 43 15 60 18 66  9  3 31 50 64 56  7 42 35 48 29 65 38 68
     32 55 37 36 41  6  4 33 67 23 57 61 14 20 53 40 59 24 44 19 54 51 26 39]
    PhoneService ['No' 'Yes']
    MultipleLines ['No' 'Yes']
    InternetService ['DSL' 'Fiber optic' 'No']
    OnlineSecurity ['No' 'Yes']
    OnlineBackup ['Yes' 'No']
    DeviceProtection ['No' 'Yes']
    TechSupport ['No' 'Yes']
    StreamingTV ['No' 'Yes']
    StreamingMovies ['No' 'Yes']
    Contract ['Month-to-month' 'One year' 'Two year']
    PaperlessBilling ['Yes' 'No']
    PaymentMethod ['Electronic check' 'Mailed check' 'Bank transfer (automatic)'
     'Credit card (automatic)']
    MonthlyCharges [29.85 56.95 53.85 ... 63.1  44.2  78.7 ]
    TotalCharges [  29.85 1889.5   108.15 ...  346.45  306.6  6844.5 ]
    Churn ['No' 'Yes']



```python
cat_cols = data.select_dtypes('object').columns
```


```python
for i in cat_cols:
    if len(data[i].unique()) ==2: 
        data[i] = lb.fit_transform(data[i])
```


```python
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
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>DSL</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Month-to-month</td>
      <td>1</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>1</td>
      <td>0</td>
      <td>DSL</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>One year</td>
      <td>0</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>1889.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>DSL</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Month-to-month</td>
      <td>1</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>DSL</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>One year</td>
      <td>0</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>Fiber optic</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Month-to-month</td>
      <td>1</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_data = pd.get_dummies(data,columns=['InternetService','Contract','PaymentMethod'])
```


```python
new_data.head()
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
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>...</th>
      <th>InternetService_DSL</th>
      <th>InternetService_Fiber optic</th>
      <th>InternetService_No</th>
      <th>Contract_Month-to-month</th>
      <th>Contract_One year</th>
      <th>Contract_Two year</th>
      <th>PaymentMethod_Bank transfer (automatic)</th>
      <th>PaymentMethod_Credit card (automatic)</th>
      <th>PaymentMethod_Electronic check</th>
      <th>PaymentMethod_Mailed check</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
scale  = ['tenure','MonthlyCharges','TotalCharges']
from sklearn.preprocessing import MinMaxScaler

mc = MinMaxScaler()
```


```python
new_data[scale] = mc.fit_transform(new_data[scale])
```


```python
new_data.head()
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
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>...</th>
      <th>InternetService_DSL</th>
      <th>InternetService_Fiber optic</th>
      <th>InternetService_No</th>
      <th>Contract_Month-to-month</th>
      <th>Contract_One year</th>
      <th>Contract_Two year</th>
      <th>PaymentMethod_Bank transfer (automatic)</th>
      <th>PaymentMethod_Credit card (automatic)</th>
      <th>PaymentMethod_Electronic check</th>
      <th>PaymentMethod_Mailed check</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.464789</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.014085</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.619718</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.014085</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
new_data.shape
```




    (7032, 27)




```python
X = new_data.drop('Churn',axis='columns')
```


```python
Y = new_data['Churn']
```


```python
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,random_state=5)
```


```python
X_train.shape
```




    (5625, 26)




```python
X_test.shape
```




    (1407, 26)




```python
model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=100)
```

    Epoch 1/100
    176/176 [==============================] - 0s 861us/step - loss: 0.5007 - accuracy: 0.7444
    Epoch 2/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.4304 - accuracy: 0.7963
    Epoch 3/100
    176/176 [==============================] - 0s 690us/step - loss: 0.4190 - accuracy: 0.8021
    Epoch 4/100
    176/176 [==============================] - 0s 668us/step - loss: 0.4141 - accuracy: 0.8085
    Epoch 5/100
    176/176 [==============================] - 0s 617us/step - loss: 0.4113 - accuracy: 0.8087
    Epoch 6/100
    176/176 [==============================] - 0s 660us/step - loss: 0.4082 - accuracy: 0.8084
    Epoch 7/100
    176/176 [==============================] - 0s 2ms/step - loss: 0.4065 - accuracy: 0.8107
    Epoch 8/100
    176/176 [==============================] - 0s 986us/step - loss: 0.4037 - accuracy: 0.8117
    Epoch 9/100
    176/176 [==============================] - 0s 2ms/step - loss: 0.4030 - accuracy: 0.8149
    Epoch 10/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.4010 - accuracy: 0.8155
    ...
    Epoch 90/100
    176/176 [==============================] - 0s 809us/step - loss: 0.3533 - accuracy: 0.8334
    Epoch 91/100
    176/176 [==============================] - 0s 785us/step - loss: 0.3529 - accuracy: 0.8359
    Epoch 92/100
    176/176 [==============================] - 0s 788us/step - loss: 0.3530 - accuracy: 0.8336
    Epoch 93/100
    176/176 [==============================] - 0s 678us/step - loss: 0.3525 - accuracy: 0.8354
    Epoch 94/100
    176/176 [==============================] - 0s 701us/step - loss: 0.3521 - accuracy: 0.8315
    Epoch 95/100
    176/176 [==============================] - 0s 637us/step - loss: 0.3524 - accuracy: 0.8350
    Epoch 96/100
    176/176 [==============================] - 0s 627us/step - loss: 0.3526 - accuracy: 0.8325
    Epoch 97/100
    176/176 [==============================] - 0s 624us/step - loss: 0.3526 - accuracy: 0.8356
    Epoch 98/100
    176/176 [==============================] - 0s 618us/step - loss: 0.3498 - accuracy: 0.8347
    Epoch 99/100
    176/176 [==============================] - 0s 631us/step - loss: 0.3514 - accuracy: 0.8322
    Epoch 100/100
    176/176 [==============================] - 0s 792us/step - loss: 0.3491 - accuracy: 0.8350





    <tensorflow.python.keras.callbacks.History at 0x7fe907617dd0>




```python
model.evaluate(X_test,Y_test)
```

    44/44 [==============================] - 0s 655us/step - loss: 0.4698 - accuracy: 0.7818





    [0.46982309222221375, 0.7818052768707275]




```python
y_pred = model.predict(X_test)
y_pred
```




    array([[0.34011692],
           [0.52507246],
           [0.00873426],
           ...,
           [0.7583121 ],
           [0.6415251 ],
           [0.6426179 ]], dtype=float32)




```python
def convert_values(values):
    y_pred = []
    for i in values:
        if i > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred
```


```python
y_pred = convert_values(y_pred)
```


```python
from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(Y_test,y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.82      0.88      0.85       999
               1       0.65      0.53      0.59       408
    
        accuracy                           0.78      1407
       macro avg       0.74      0.71      0.72      1407
    weighted avg       0.77      0.78      0.77      1407
    



```python
import seaborn as sn
cm = confusion_matrix(Y_test,y_pred)
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
```




    Text(69.0, 0.5, 'Truth')




![png](/assets/img/sample/output_53_1.png)



```python
cm
```




    array([[882, 117],
           [190, 218]])



Trying out Dropout to improve performance on Test set.


```python
modeld = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])


modeld.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

modeld.fit(X_train, Y_train, epochs=100)
```

    Epoch 1/100
    176/176 [==============================] - 0s 661us/step - loss: 0.5065 - accuracy: 0.7438
    Epoch 2/100
    176/176 [==============================] - 0s 637us/step - loss: 0.4495 - accuracy: 0.7723
    Epoch 3/100
    176/176 [==============================] - 0s 671us/step - loss: 0.4457 - accuracy: 0.7840
    Epoch 4/100
    176/176 [==============================] - 0s 663us/step - loss: 0.4384 - accuracy: 0.7854
    Epoch 5/100
    176/176 [==============================] - 0s 816us/step - loss: 0.4342 - accuracy: 0.7915
    Epoch 6/100
    176/176 [==============================] - 0s 633us/step - loss: 0.4311 - accuracy: 0.7979
    Epoch 7/100
    176/176 [==============================] - 0s 643us/step - loss: 0.4254 - accuracy: 0.7988
    Epoch 8/100
    176/176 [==============================] - 0s 764us/step - loss: 0.4237 - accuracy: 0.7998
    Epoch 9/100
    176/176 [==============================] - 0s 819us/step - loss: 0.4228 - accuracy: 0.7993
    Epoch 10/100
    176/176 [==============================] - 0s 890us/step - loss: 0.4198 - accuracy: 0.7980
    ...
    Epoch 91/100
    176/176 [==============================] - 0s 616us/step - loss: 0.3814 - accuracy: 0.8199
    Epoch 92/100
    176/176 [==============================] - 0s 770us/step - loss: 0.3849 - accuracy: 0.8171
    Epoch 93/100
    176/176 [==============================] - 0s 633us/step - loss: 0.3818 - accuracy: 0.8220
    Epoch 94/100
    176/176 [==============================] - 0s 675us/step - loss: 0.3821 - accuracy: 0.8219
    Epoch 95/100
    176/176 [==============================] - 0s 607us/step - loss: 0.3760 - accuracy: 0.8252
    Epoch 96/100
    176/176 [==============================] - 0s 606us/step - loss: 0.3827 - accuracy: 0.8224
    Epoch 97/100
    176/176 [==============================] - 0s 623us/step - loss: 0.3811 - accuracy: 0.8210
    Epoch 98/100
    176/176 [==============================] - 0s 604us/step - loss: 0.3784 - accuracy: 0.8252
    Epoch 99/100
    176/176 [==============================] - 0s 610us/step - loss: 0.3784 - accuracy: 0.8228
    Epoch 100/100
    176/176 [==============================] - 0s 596us/step - loss: 0.3773 - accuracy: 0.8220





    <tensorflow.python.keras.callbacks.History at 0x7fe8e86a3bd0>




```python
modeld.evaluate(X_test,Y_test)
```

    44/44 [==============================] - 0s 1ms/step - loss: 0.4618 - accuracy: 0.7740





    [0.4618297219276428, 0.7739872336387634]




```python
y_predd = modeld.predict(X_test)
y_predd
```




    array([[0.38572046],
           [0.44322282],
           [0.0048472 ],
           ...,
           [0.7393723 ],
           [0.68880296],
           [0.75985956]], dtype=float32)




```python
y_predd = convert_values(y_predd)
```


```python
print(classification_report(Y_test,y_predd))
```

                  precision    recall  f1-score   support
    
               0       0.81      0.88      0.85       999
               1       0.64      0.50      0.56       408
    
        accuracy                           0.77      1407
       macro avg       0.73      0.69      0.71      1407
    weighted avg       0.76      0.77      0.77      1407
    


We can see that accuracy is improved. There is slight increase in F1-score for both the classes. But, the F-1 score of class 1 is still not good as compared to class 0. We need to handle this imbalance dataset to improve our model. There are different ways to handle this such as 
- Undersampling your majority class
- Oversampling your minority classes by dulpication
- Using SMOTE (Synthetic Minority Over-Sampling Technique)
- Generate Synthetic examples using KNN method. 

There is another one way to handle this - Focal Loss. What Focal Loss is? It penalizes majority samples during loss calculation and give more weight to minority class samples. Lets see an implementation to overcome this situation.

## Undersampling


```python
yes_class_data = new_data[new_data['Churn']==1]
no_class_data = new_data[new_data['Churn']==0]
```


```python
yes_class_data.shape
```




    (1869, 27)




```python
no_class_data.shape
```




    (5163, 27)




```python
no_class_data_under = no_class_data.sample(yes_class_data.shape[0])
```


```python
no_class_data_under.shape
```




    (1869, 27)



As we can see that, now we have 1869 samples of class 0. We have reduced our number of samples from 5163 to 1869. We will combine these two datasets and build our model.


```python
under_df = pd.concat([no_class_data_under,yes_class_data],axis=0)
under_df.shape
```




    (3738, 27)




```python
under_df['Churn'].value_counts()
```




    1    1869
    0    1869
    Name: Churn, dtype: int64




```python
X_under = under_df.drop('Churn',axis='columns')
y_under = under_df['Churn']
```


```python
X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X_under, y_under, test_size=0.2, random_state=15, stratify=y_under)
```

Here, we used stratify while splitting our data. What it does? It maintains the proportion of both the classes in training set as well as test set.


```python
y_train_u.value_counts()
```




    1    1495
    0    1495
    Name: Churn, dtype: int64




```python
model_u = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])


model_u.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_u.fit(X_train_u, y_train_u, epochs=100)
```

    Epoch 1/100
    94/94 [==============================] - 0s 1ms/step - loss: 0.5855 - accuracy: 0.6629
    Epoch 2/100
    94/94 [==============================] - 0s 905us/step - loss: 0.5077 - accuracy: 0.7555
    Epoch 3/100
    94/94 [==============================] - 0s 1ms/step - loss: 0.4938 - accuracy: 0.7555
    Epoch 4/100
    94/94 [==============================] - 0s 685us/step - loss: 0.4866 - accuracy: 0.7662
    Epoch 5/100
    94/94 [==============================] - 0s 727us/step - loss: 0.4834 - accuracy: 0.7642
    Epoch 6/100
    94/94 [==============================] - 0s 946us/step - loss: 0.4798 - accuracy: 0.7689
    Epoch 7/100
    94/94 [==============================] - 0s 641us/step - loss: 0.4772 - accuracy: 0.7712
    Epoch 8/100
    94/94 [==============================] - 0s 621us/step - loss: 0.4745 - accuracy: 0.7756
    Epoch 9/100
    94/94 [==============================] - 0s 710us/step - loss: 0.4727 - accuracy: 0.7753
    Epoch 10/100
    94/94 [==============================] - 0s 727us/step - loss: 0.4724 - accuracy: 0.7773
    ...
    Epoch 91/100
    94/94 [==============================] - 0s 717us/step - loss: 0.3769 - accuracy: 0.8298
    Epoch 92/100
    94/94 [==============================] - 0s 712us/step - loss: 0.3776 - accuracy: 0.8304
    Epoch 93/100
    94/94 [==============================] - 0s 716us/step - loss: 0.3807 - accuracy: 0.8278
    Epoch 94/100
    94/94 [==============================] - 0s 695us/step - loss: 0.3770 - accuracy: 0.8308
    Epoch 95/100
    94/94 [==============================] - 0s 753us/step - loss: 0.3754 - accuracy: 0.8341
    Epoch 96/100
    94/94 [==============================] - 0s 791us/step - loss: 0.3761 - accuracy: 0.8241
    Epoch 97/100
    94/94 [==============================] - 0s 644us/step - loss: 0.3726 - accuracy: 0.8311
    Epoch 98/100
    94/94 [==============================] - 0s 596us/step - loss: 0.3745 - accuracy: 0.8318
    Epoch 99/100
    94/94 [==============================] - 0s 589us/step - loss: 0.3748 - accuracy: 0.8334
    Epoch 100/100
    94/94 [==============================] - 0s 2ms/step - loss: 0.3716 - accuracy: 0.8281





    <tensorflow.python.keras.callbacks.History at 0x7fe8e9359d10>




```python
model_u.evaluate(X_test_u,y_test_u)
```

    24/24 [==============================] - 0s 526us/step - loss: 0.6064 - accuracy: 0.7299





    [0.6063751578330994, 0.7299465537071228]




```python
y_pred_u = model_u.predict(X_test_u)
y_pred_u
```




    array([[5.69502771e-01],
           [9.16280985e-01],
           [9.82213259e-01],
           [8.72887850e-01],
           [2.06746757e-01],
           [6.96271479e-01],
           [6.68895245e-03],
           [8.05474699e-01],
           [7.27444410e-01],
            ...
           [5.10366023e-01],
           [9.41771984e-01],
           [5.27670979e-03],
           [6.11849427e-01],
           [3.80265623e-01],
           [2.91931033e-02],
           [9.59493160e-01],
           [8.60638499e-01],
           [3.00275683e-02],
           [3.53098392e-01],
           [4.03191745e-01]], dtype=float32)




```python
y_pred_u = convert_values(y_pred_u)
```


```python
print(classification_report(y_test_u,y_pred_u))
```

                  precision    recall  f1-score   support
    
               0       0.73      0.72      0.73       374
               1       0.73      0.74      0.73       374
    
        accuracy                           0.73       748
       macro avg       0.73      0.73      0.73       748
    weighted avg       0.73      0.73      0.73       748
    


Now, we can see the results. The F1-score for minority class 1 has improved Significantly. But, Score for class 0 got reduced because we reduced samples of this class and it is expected. So, We have more generalized classifier which classifies both classes with similar prediction score. Now, We will implement Oversampling method by generating duplicate samples.

## Oversampling


```python
yes_class_data_over = yes_class_data.sample(no_class_data.shape[0],replace=True)
yes_class_data_over.shape
```




    (5163, 27)



We have got more samples of class 1. Lets combine these two classes and implement our model.


```python
over_df = pd.concat([no_class_data,yes_class_data_over],axis=0)
over_df.shape
```




    (10326, 27)




```python
over_df['Churn'].value_counts()
```




    1    5163
    0    5163
    Name: Churn, dtype: int64




```python
X_o = over_df.drop('Churn',axis='columns')
y_o = over_df['Churn']

X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_o, y_o, test_size=0.2, random_state=15, stratify=y_o)
```


```python
model_o = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])


model_o.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_o.fit(X_train_o, y_train_o, epochs=100)
```

    Epoch 1/100
    259/259 [==============================] - 0s 812us/step - loss: 0.5566 - accuracy: 0.7230
    Epoch 2/100
    259/259 [==============================] - 0s 752us/step - loss: 0.4876 - accuracy: 0.7663
    Epoch 3/100
    259/259 [==============================] - 0s 573us/step - loss: 0.4785 - accuracy: 0.7683
    Epoch 4/100
    259/259 [==============================] - 0s 1ms/step - loss: 0.4745 - accuracy: 0.7691
    Epoch 5/100
    259/259 [==============================] - 0s 1ms/step - loss: 0.4710 - accuracy: 0.7723
    Epoch 6/100
    259/259 [==============================] - 0s 649us/step - loss: 0.4690 - accuracy: 0.7755
    Epoch 7/100
    259/259 [==============================] - 0s 653us/step - loss: 0.4657 - accuracy: 0.7771
    Epoch 8/100
    259/259 [==============================] - 0s 730us/step - loss: 0.4634 - accuracy: 0.7774
    Epoch 9/100
    259/259 [==============================] - 0s 870us/step - loss: 0.4603 - accuracy: 0.7821
    Epoch 10/100
    259/259 [==============================] - 0s 772us/step - loss: 0.4587 - accuracy: 0.7837
    ...
    Epoch 91/100
    259/259 [==============================] - 0s 679us/step - loss: 0.3571 - accuracy: 0.8391
    Epoch 92/100
    259/259 [==============================] - 0s 692us/step - loss: 0.3563 - accuracy: 0.8379
    Epoch 93/100
    259/259 [==============================] - 0s 766us/step - loss: 0.3564 - accuracy: 0.8416
    Epoch 94/100
    259/259 [==============================] - 0s 874us/step - loss: 0.3556 - accuracy: 0.8409
    Epoch 95/100
    259/259 [==============================] - 0s 815us/step - loss: 0.3539 - accuracy: 0.8416
    Epoch 96/100
    259/259 [==============================] - 0s 641us/step - loss: 0.3534 - accuracy: 0.8418
    Epoch 97/100
    259/259 [==============================] - 0s 614us/step - loss: 0.3536 - accuracy: 0.8408
    Epoch 98/100
    259/259 [==============================] - 0s 1ms/step - loss: 0.3523 - accuracy: 0.8416
    Epoch 99/100
    259/259 [==============================] - 0s 941us/step - loss: 0.3533 - accuracy: 0.8406
    Epoch 100/100
    259/259 [==============================] - 0s 630us/step - loss: 0.3513 - accuracy: 0.8446





    <tensorflow.python.keras.callbacks.History at 0x7fe9063c5a50>




```python
model_o.evaluate(X_test_o,y_test_o)
```

    65/65 [==============================] - 0s 539us/step - loss: 0.4639 - accuracy: 0.7904





    [0.46388959884643555, 0.7904162406921387]




```python
y_pred_o = model_o.predict(X_test_o)
y_pred_o = convert_values(y_pred_o)
print(classification_report(y_test_o,y_pred_o))
```

                  precision    recall  f1-score   support
    
               0       0.82      0.74      0.78      1033
               1       0.76      0.84      0.80      1033
    
        accuracy                           0.79      2066
       macro avg       0.79      0.79      0.79      2066
    weighted avg       0.79      0.79      0.79      2066
    


We have got appoximately the same f1-score for both the classes. Our model is generalizing well. This is not actually a good method as it just copies the data and creates duplicate samples. Now, we will see another important and widely used technique to handle imabalanced dataset - SMOTE.

## SMOTE


```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_sample(X, Y)

y_sm.value_counts()
```




    1    5163
    0    5163
    Name: Churn, dtype: int64




```python
X_train_s,X_test_s,Y_train_s,Y_test_s = train_test_split(X_sm,y_sm, test_size=0.2,random_state=15,stratify=y_sm)
```


```python
model_s = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])


model_s.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_s.fit(X_train_s, Y_train_s, epochs=100)
```

    Epoch 1/100
    259/259 [==============================] - 0s 622us/step - loss: 0.5234 - accuracy: 0.7479
    Epoch 2/100
    259/259 [==============================] - 0s 668us/step - loss: 0.4734 - accuracy: 0.7785
    Epoch 3/100
    259/259 [==============================] - 0s 647us/step - loss: 0.4638 - accuracy: 0.7806
    Epoch 4/100
    259/259 [==============================] - 0s 618us/step - loss: 0.4572 - accuracy: 0.7852
    Epoch 5/100
    259/259 [==============================] - 0s 604us/step - loss: 0.4514 - accuracy: 0.7904
    Epoch 6/100
    259/259 [==============================] - 0s 606us/step - loss: 0.4456 - accuracy: 0.7936
    Epoch 7/100
    259/259 [==============================] - 0s 598us/step - loss: 0.4406 - accuracy: 0.7983
    Epoch 8/100
    259/259 [==============================] - 0s 599us/step - loss: 0.4372 - accuracy: 0.7983
    Epoch 9/100
    259/259 [==============================] - 0s 598us/step - loss: 0.4333 - accuracy: 0.8036
    Epoch 10/100
    259/259 [==============================] - 0s 626us/step - loss: 0.4298 - accuracy: 0.8040
    ...
    Epoch 91/100
    259/259 [==============================] - 0s 662us/step - loss: 0.3405 - accuracy: 0.8551
    Epoch 92/100
    259/259 [==============================] - 0s 651us/step - loss: 0.3429 - accuracy: 0.8522
    Epoch 93/100
    259/259 [==============================] - 0s 696us/step - loss: 0.3409 - accuracy: 0.8498
    Epoch 94/100
    259/259 [==============================] - 0s 691us/step - loss: 0.3409 - accuracy: 0.8531
    Epoch 95/100
    259/259 [==============================] - 0s 656us/step - loss: 0.3404 - accuracy: 0.8518
    Epoch 96/100
    259/259 [==============================] - 0s 876us/step - loss: 0.3405 - accuracy: 0.8524
    Epoch 97/100
    259/259 [==============================] - 0s 772us/step - loss: 0.3390 - accuracy: 0.8523
    Epoch 98/100
    259/259 [==============================] - 0s 755us/step - loss: 0.3402 - accuracy: 0.8554
    Epoch 99/100
    259/259 [==============================] - 0s 707us/step - loss: 0.3401 - accuracy: 0.8519
    Epoch 100/100
    259/259 [==============================] - 0s 629us/step - loss: 0.3381 - accuracy: 0.8553





    <tensorflow.python.keras.callbacks.History at 0x7fe8eb3e9f10>




```python
model_s.evaluate(X_test_s,Y_test_s)
```

    65/65 [==============================] - 0s 601us/step - loss: 0.4381 - accuracy: 0.8025





    [0.4381003677845001, 0.8025169372558594]




```python
y_pred_s = model_s.predict(X_test_s)
y_pred_s = convert_values(y_pred_s)
print(classification_report(Y_test_s,y_pred_s))
```

                  precision    recall  f1-score   support
    
               0       0.87      0.72      0.78      1033
               1       0.76      0.89      0.82      1033
    
        accuracy                           0.80      2066
       macro avg       0.81      0.80      0.80      2066
    weighted avg       0.81      0.80      0.80      2066
    


SMOTE is giving out the best results so far. Precision and recall for respective classes are consistent.


```python
from focal_loss import BinaryFocalLoss
```


```python

model = tf.keras.Model(...)
model.compile(
    optimizer=...,
    loss=BinaryFocalLoss(gamma=2),  # Used here like a tf.keras loss
    metrics=...,
)
history = model.fit(...)
```


```python
model_f = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])


model_f.compile(optimizer='adam',
              loss=BinaryFocalLoss(gamma=2),
              metrics=['accuracy'])

model_f.fit(X_train, Y_train, epochs=100)
```

    Epoch 1/100
    176/176 [==============================] - 0s 838us/step - loss: 0.1222 - accuracy: 0.7682
    Epoch 2/100
    176/176 [==============================] - 0s 911us/step - loss: 0.1112 - accuracy: 0.7943
    Epoch 3/100
    176/176 [==============================] - 0s 800us/step - loss: 0.1091 - accuracy: 0.7979
    Epoch 4/100
    176/176 [==============================] - 0s 673us/step - loss: 0.1080 - accuracy: 0.8011
    Epoch 5/100
    176/176 [==============================] - 0s 662us/step - loss: 0.1069 - accuracy: 0.8032
    Epoch 6/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.1067 - accuracy: 0.8000
    Epoch 7/100
    176/176 [==============================] - 0s 970us/step - loss: 0.1061 - accuracy: 0.8039
    Epoch 8/100
    176/176 [==============================] - 0s 990us/step - loss: 0.1056 - accuracy: 0.8084
    Epoch 9/100
    176/176 [==============================] - 0s 896us/step - loss: 0.1045 - accuracy: 0.8103
    Epoch 10/100
    176/176 [==============================] - 0s 943us/step - loss: 0.1041 - accuracy: 0.8130
    ...
    Epoch 91/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.0854 - accuracy: 0.8382
    Epoch 92/100
    176/176 [==============================] - 0s 719us/step - loss: 0.0851 - accuracy: 0.8370
    Epoch 93/100
    176/176 [==============================] - 0s 945us/step - loss: 0.0856 - accuracy: 0.8364
    Epoch 94/100
    176/176 [==============================] - 0s 791us/step - loss: 0.0855 - accuracy: 0.8388
    Epoch 95/100
    176/176 [==============================] - 0s 850us/step - loss: 0.0852 - accuracy: 0.8391
    Epoch 96/100
    176/176 [==============================] - 0s 702us/step - loss: 0.0849 - accuracy: 0.8386
    Epoch 97/100
    176/176 [==============================] - 0s 894us/step - loss: 0.0850 - accuracy: 0.8412
    Epoch 98/100
    176/176 [==============================] - 0s 856us/step - loss: 0.0845 - accuracy: 0.8414
    Epoch 99/100
    176/176 [==============================] - 0s 800us/step - loss: 0.0844 - accuracy: 0.8389
    Epoch 100/100
    176/176 [==============================] - 0s 758us/step - loss: 0.0844 - accuracy: 0.8412





    <tensorflow.python.keras.callbacks.History at 0x7fe8ec46d750>




```python
model_f.evaluate(X_test,Y_test)
```

    44/44 [==============================] - 0s 821us/step - loss: 0.1431 - accuracy: 0.7662





    [0.14313486218452454, 0.7661691308021545]




```python
y_pred_f = model_f.predict(X_test)
y_pred_f = convert_values(y_pred_f)
print(classification_report(Y_test,y_pred_f))
```

                  precision    recall  f1-score   support
    
               0       0.84      0.82      0.83       999
               1       0.59      0.63      0.61       408
    
        accuracy                           0.77      1407
       macro avg       0.72      0.73      0.72      1407
    weighted avg       0.77      0.77      0.77      1407
    


We can see that the F-1 score is not as good as it is for class 0. Focal Loss is not helping much in overcoming the imbalanced datasets. It works effectively in Object detection where we errorneously identity objects in emply spaces.  

So, we can conclude that, SMOTE is very effective in handling imbalanced datasets as it generates synthetic data samples and prevents the problem of overfitting.


```python

```
