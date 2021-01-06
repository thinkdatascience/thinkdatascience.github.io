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
    Epoch 11/100
    176/176 [==============================] - 0s 2ms/step - loss: 0.4004 - accuracy: 0.8130
    Epoch 12/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.3977 - accuracy: 0.8146
    Epoch 13/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.3968 - accuracy: 0.8171
    Epoch 14/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.3947 - accuracy: 0.8156
    Epoch 15/100
    176/176 [==============================] - 0s 792us/step - loss: 0.3936 - accuracy: 0.8174
    Epoch 16/100
    176/176 [==============================] - 0s 685us/step - loss: 0.3933 - accuracy: 0.8180
    Epoch 17/100
    176/176 [==============================] - 0s 747us/step - loss: 0.3917 - accuracy: 0.8183
    Epoch 18/100
    176/176 [==============================] - 0s 757us/step - loss: 0.3911 - accuracy: 0.8180
    Epoch 19/100
    176/176 [==============================] - 0s 742us/step - loss: 0.3901 - accuracy: 0.8176
    Epoch 20/100
    176/176 [==============================] - 0s 931us/step - loss: 0.3886 - accuracy: 0.8203
    Epoch 21/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.3883 - accuracy: 0.8201
    Epoch 22/100
    176/176 [==============================] - 0s 790us/step - loss: 0.3868 - accuracy: 0.8206
    Epoch 23/100
    176/176 [==============================] - 0s 686us/step - loss: 0.3869 - accuracy: 0.8192
    Epoch 24/100
    176/176 [==============================] - 0s 724us/step - loss: 0.3857 - accuracy: 0.8212
    Epoch 25/100
    176/176 [==============================] - 0s 674us/step - loss: 0.3842 - accuracy: 0.8199
    Epoch 26/100
    176/176 [==============================] - 0s 758us/step - loss: 0.3835 - accuracy: 0.8201
    Epoch 27/100
    176/176 [==============================] - 0s 667us/step - loss: 0.3823 - accuracy: 0.8213
    Epoch 28/100
    176/176 [==============================] - 0s 751us/step - loss: 0.3822 - accuracy: 0.8226
    Epoch 29/100
    176/176 [==============================] - 0s 708us/step - loss: 0.3816 - accuracy: 0.8201
    Epoch 30/100
    176/176 [==============================] - 0s 694us/step - loss: 0.3816 - accuracy: 0.8204
    Epoch 31/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.3815 - accuracy: 0.8192
    Epoch 32/100
    176/176 [==============================] - 0s 699us/step - loss: 0.3806 - accuracy: 0.8199
    Epoch 33/100
    176/176 [==============================] - 0s 643us/step - loss: 0.3795 - accuracy: 0.8226
    Epoch 34/100
    176/176 [==============================] - 0s 690us/step - loss: 0.3804 - accuracy: 0.8238
    Epoch 35/100
    176/176 [==============================] - 0s 613us/step - loss: 0.3795 - accuracy: 0.8212
    Epoch 36/100
    176/176 [==============================] - 0s 902us/step - loss: 0.3784 - accuracy: 0.8249
    Epoch 37/100
    176/176 [==============================] - 0s 1000us/step - loss: 0.3771 - accuracy: 0.8238
    Epoch 38/100
    176/176 [==============================] - 0s 966us/step - loss: 0.3764 - accuracy: 0.8238
    Epoch 39/100
    176/176 [==============================] - 0s 926us/step - loss: 0.3761 - accuracy: 0.8236
    Epoch 40/100
    176/176 [==============================] - 0s 869us/step - loss: 0.3765 - accuracy: 0.8242
    Epoch 41/100
    176/176 [==============================] - 0s 797us/step - loss: 0.3753 - accuracy: 0.8261
    Epoch 42/100
    176/176 [==============================] - 0s 889us/step - loss: 0.3740 - accuracy: 0.8252
    Epoch 43/100
    176/176 [==============================] - 0s 741us/step - loss: 0.3746 - accuracy: 0.8235
    Epoch 44/100
    176/176 [==============================] - 0s 887us/step - loss: 0.3734 - accuracy: 0.8252
    Epoch 45/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.3732 - accuracy: 0.8236
    Epoch 46/100
    176/176 [==============================] - 0s 975us/step - loss: 0.3723 - accuracy: 0.8245
    Epoch 47/100
    176/176 [==============================] - 0s 759us/step - loss: 0.3731 - accuracy: 0.8229
    Epoch 48/100
    176/176 [==============================] - 0s 610us/step - loss: 0.3716 - accuracy: 0.8242
    Epoch 49/100
    176/176 [==============================] - 0s 598us/step - loss: 0.3720 - accuracy: 0.8256
    Epoch 50/100
    176/176 [==============================] - 0s 679us/step - loss: 0.3721 - accuracy: 0.8252
    Epoch 51/100
    176/176 [==============================] - 0s 797us/step - loss: 0.3708 - accuracy: 0.8247
    Epoch 52/100
    176/176 [==============================] - 0s 696us/step - loss: 0.3704 - accuracy: 0.8267
    Epoch 53/100
    176/176 [==============================] - 0s 707us/step - loss: 0.3697 - accuracy: 0.8286
    Epoch 54/100
    176/176 [==============================] - 0s 587us/step - loss: 0.3696 - accuracy: 0.8265
    Epoch 55/100
    176/176 [==============================] - 0s 578us/step - loss: 0.3701 - accuracy: 0.8244
    Epoch 56/100
    176/176 [==============================] - 0s 591us/step - loss: 0.3685 - accuracy: 0.8254
    Epoch 57/100
    176/176 [==============================] - 0s 786us/step - loss: 0.3673 - accuracy: 0.8260
    Epoch 58/100
    176/176 [==============================] - 0s 698us/step - loss: 0.3674 - accuracy: 0.8276
    Epoch 59/100
    176/176 [==============================] - 0s 596us/step - loss: 0.3672 - accuracy: 0.8295
    Epoch 60/100
    176/176 [==============================] - 0s 628us/step - loss: 0.3670 - accuracy: 0.8277
    Epoch 61/100
    176/176 [==============================] - 0s 606us/step - loss: 0.3664 - accuracy: 0.8286
    Epoch 62/100
    176/176 [==============================] - 0s 588us/step - loss: 0.3662 - accuracy: 0.8295
    Epoch 63/100
    176/176 [==============================] - 0s 591us/step - loss: 0.3653 - accuracy: 0.8302
    Epoch 64/100
    176/176 [==============================] - 0s 579us/step - loss: 0.3655 - accuracy: 0.8290
    Epoch 65/100
    176/176 [==============================] - 0s 593us/step - loss: 0.3646 - accuracy: 0.8300
    Epoch 66/100
    176/176 [==============================] - 0s 586us/step - loss: 0.3636 - accuracy: 0.8302
    Epoch 67/100
    176/176 [==============================] - 0s 574us/step - loss: 0.3630 - accuracy: 0.8290
    Epoch 68/100
    176/176 [==============================] - 0s 646us/step - loss: 0.3631 - accuracy: 0.8324
    Epoch 69/100
    176/176 [==============================] - 0s 602us/step - loss: 0.3628 - accuracy: 0.8284
    Epoch 70/100
    176/176 [==============================] - 0s 643us/step - loss: 0.3623 - accuracy: 0.8318
    Epoch 71/100
    176/176 [==============================] - 0s 615us/step - loss: 0.3624 - accuracy: 0.8299
    Epoch 72/100
    176/176 [==============================] - 0s 589us/step - loss: 0.3602 - accuracy: 0.8309
    Epoch 73/100
    176/176 [==============================] - 0s 579us/step - loss: 0.3619 - accuracy: 0.8304
    Epoch 74/100
    176/176 [==============================] - 0s 683us/step - loss: 0.3600 - accuracy: 0.8308
    Epoch 75/100
    176/176 [==============================] - 0s 970us/step - loss: 0.3606 - accuracy: 0.8309
    Epoch 76/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.3601 - accuracy: 0.8300
    Epoch 77/100
    176/176 [==============================] - 0s 832us/step - loss: 0.3598 - accuracy: 0.8315
    Epoch 78/100
    176/176 [==============================] - 0s 746us/step - loss: 0.3588 - accuracy: 0.8300
    Epoch 79/100
    176/176 [==============================] - 0s 741us/step - loss: 0.3594 - accuracy: 0.8318
    Epoch 80/100
    176/176 [==============================] - 0s 792us/step - loss: 0.3584 - accuracy: 0.8297
    Epoch 81/100
    176/176 [==============================] - 0s 732us/step - loss: 0.3569 - accuracy: 0.8356
    Epoch 82/100
    176/176 [==============================] - 0s 2ms/step - loss: 0.3579 - accuracy: 0.8332
    Epoch 83/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.3571 - accuracy: 0.8336
    Epoch 84/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.3563 - accuracy: 0.8338
    Epoch 85/100
    176/176 [==============================] - 0s 669us/step - loss: 0.3558 - accuracy: 0.8292
    Epoch 86/100
    176/176 [==============================] - 0s 676us/step - loss: 0.3545 - accuracy: 0.8336
    Epoch 87/100
    176/176 [==============================] - 0s 788us/step - loss: 0.3545 - accuracy: 0.8332
    Epoch 88/100
    176/176 [==============================] - 0s 2ms/step - loss: 0.3559 - accuracy: 0.8322
    Epoch 89/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.3540 - accuracy: 0.8318
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
    Epoch 11/100
    176/176 [==============================] - 0s 886us/step - loss: 0.4191 - accuracy: 0.8023
    Epoch 12/100
    176/176 [==============================] - 0s 927us/step - loss: 0.4195 - accuracy: 0.8041
    Epoch 13/100
    176/176 [==============================] - 0s 892us/step - loss: 0.4146 - accuracy: 0.8073
    Epoch 14/100
    176/176 [==============================] - 0s 856us/step - loss: 0.4188 - accuracy: 0.8037
    Epoch 15/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.4153 - accuracy: 0.8060
    Epoch 16/100
    176/176 [==============================] - 0s 779us/step - loss: 0.4134 - accuracy: 0.8082
    Epoch 17/100
    176/176 [==============================] - 0s 722us/step - loss: 0.4150 - accuracy: 0.8043
    Epoch 18/100
    176/176 [==============================] - 0s 757us/step - loss: 0.4111 - accuracy: 0.8087
    Epoch 19/100
    176/176 [==============================] - 0s 736us/step - loss: 0.4128 - accuracy: 0.8105
    Epoch 20/100
    176/176 [==============================] - 0s 736us/step - loss: 0.4141 - accuracy: 0.8062
    Epoch 21/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.4149 - accuracy: 0.8076
    Epoch 22/100
    176/176 [==============================] - 0s 968us/step - loss: 0.4090 - accuracy: 0.8084
    Epoch 23/100
    176/176 [==============================] - 0s 995us/step - loss: 0.4076 - accuracy: 0.8105
    Epoch 24/100
    176/176 [==============================] - 0s 977us/step - loss: 0.4081 - accuracy: 0.8087
    Epoch 25/100
    176/176 [==============================] - 0s 994us/step - loss: 0.4086 - accuracy: 0.8092
    Epoch 26/100
    176/176 [==============================] - 0s 919us/step - loss: 0.4080 - accuracy: 0.8071
    Epoch 27/100
    176/176 [==============================] - 0s 846us/step - loss: 0.4054 - accuracy: 0.8082
    Epoch 28/100
    176/176 [==============================] - 0s 864us/step - loss: 0.4061 - accuracy: 0.8076
    Epoch 29/100
    176/176 [==============================] - 0s 837us/step - loss: 0.4070 - accuracy: 0.8114
    Epoch 30/100
    176/176 [==============================] - 0s 839us/step - loss: 0.4039 - accuracy: 0.8092
    Epoch 31/100
    176/176 [==============================] - 0s 853us/step - loss: 0.4036 - accuracy: 0.8100
    Epoch 32/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.4018 - accuracy: 0.8110
    Epoch 33/100
    176/176 [==============================] - 0s 809us/step - loss: 0.4003 - accuracy: 0.8135
    Epoch 34/100
    176/176 [==============================] - 0s 891us/step - loss: 0.4039 - accuracy: 0.8107
    Epoch 35/100
    176/176 [==============================] - 0s 807us/step - loss: 0.4032 - accuracy: 0.8075
    Epoch 36/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.4022 - accuracy: 0.8092
    Epoch 37/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.4002 - accuracy: 0.8123
    Epoch 38/100
    176/176 [==============================] - 0s 2ms/step - loss: 0.4039 - accuracy: 0.8112
    Epoch 39/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.3995 - accuracy: 0.8133
    Epoch 40/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.3953 - accuracy: 0.8153
    Epoch 41/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.4007 - accuracy: 0.8123
    Epoch 42/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.3984 - accuracy: 0.8151
    Epoch 43/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.3950 - accuracy: 0.8139
    Epoch 44/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.4012 - accuracy: 0.8144
    Epoch 45/100
    176/176 [==============================] - 0s 708us/step - loss: 0.3997 - accuracy: 0.8126
    Epoch 46/100
    176/176 [==============================] - 0s 782us/step - loss: 0.3947 - accuracy: 0.8171
    Epoch 47/100
    176/176 [==============================] - 0s 866us/step - loss: 0.3964 - accuracy: 0.8148
    Epoch 48/100
    176/176 [==============================] - 0s 925us/step - loss: 0.3950 - accuracy: 0.8132
    Epoch 49/100
    176/176 [==============================] - 0s 817us/step - loss: 0.3975 - accuracy: 0.8140
    Epoch 50/100
    176/176 [==============================] - 0s 801us/step - loss: 0.3943 - accuracy: 0.8178
    Epoch 51/100
    176/176 [==============================] - 0s 873us/step - loss: 0.3975 - accuracy: 0.8164
    Epoch 52/100
    176/176 [==============================] - 0s 843us/step - loss: 0.3954 - accuracy: 0.8196
    Epoch 53/100
    176/176 [==============================] - 0s 815us/step - loss: 0.3902 - accuracy: 0.8156
    Epoch 54/100
    176/176 [==============================] - 0s 874us/step - loss: 0.3914 - accuracy: 0.8148
    Epoch 55/100
    176/176 [==============================] - 0s 886us/step - loss: 0.3940 - accuracy: 0.8171
    Epoch 56/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.3900 - accuracy: 0.8188
    Epoch 57/100
    176/176 [==============================] - 0s 788us/step - loss: 0.3926 - accuracy: 0.8160
    Epoch 58/100
    176/176 [==============================] - 0s 814us/step - loss: 0.3909 - accuracy: 0.8196
    Epoch 59/100
    176/176 [==============================] - 0s 751us/step - loss: 0.3904 - accuracy: 0.8176
    Epoch 60/100
    176/176 [==============================] - 0s 832us/step - loss: 0.3909 - accuracy: 0.8197
    Epoch 61/100
    176/176 [==============================] - 0s 991us/step - loss: 0.3925 - accuracy: 0.8196
    Epoch 62/100
    176/176 [==============================] - 0s 957us/step - loss: 0.3930 - accuracy: 0.8180
    Epoch 63/100
    176/176 [==============================] - 0s 835us/step - loss: 0.3875 - accuracy: 0.8210
    Epoch 64/100
    176/176 [==============================] - 0s 802us/step - loss: 0.3914 - accuracy: 0.8176
    Epoch 65/100
    176/176 [==============================] - 0s 664us/step - loss: 0.3885 - accuracy: 0.8165
    Epoch 66/100
    176/176 [==============================] - 0s 612us/step - loss: 0.3864 - accuracy: 0.8169
    Epoch 67/100
    176/176 [==============================] - 0s 625us/step - loss: 0.3875 - accuracy: 0.8204
    Epoch 68/100
    176/176 [==============================] - 0s 608us/step - loss: 0.3887 - accuracy: 0.8165
    Epoch 69/100
    176/176 [==============================] - 0s 620us/step - loss: 0.3887 - accuracy: 0.8204
    Epoch 70/100
    176/176 [==============================] - 0s 680us/step - loss: 0.3859 - accuracy: 0.8174
    Epoch 71/100
    176/176 [==============================] - 0s 618us/step - loss: 0.3822 - accuracy: 0.8240
    Epoch 72/100
    176/176 [==============================] - 0s 612us/step - loss: 0.3848 - accuracy: 0.8199
    Epoch 73/100
    176/176 [==============================] - 0s 611us/step - loss: 0.3833 - accuracy: 0.8208
    Epoch 74/100
    176/176 [==============================] - 0s 620us/step - loss: 0.3798 - accuracy: 0.8181
    Epoch 75/100
    176/176 [==============================] - 0s 606us/step - loss: 0.3886 - accuracy: 0.8219
    Epoch 76/100
    176/176 [==============================] - 0s 607us/step - loss: 0.3898 - accuracy: 0.8172
    Epoch 77/100
    176/176 [==============================] - 0s 715us/step - loss: 0.3812 - accuracy: 0.8217
    Epoch 78/100
    176/176 [==============================] - 0s 669us/step - loss: 0.3846 - accuracy: 0.8171
    Epoch 79/100
    176/176 [==============================] - 0s 615us/step - loss: 0.3846 - accuracy: 0.8231
    Epoch 80/100
    176/176 [==============================] - 0s 643us/step - loss: 0.3799 - accuracy: 0.8228
    Epoch 81/100
    176/176 [==============================] - 0s 749us/step - loss: 0.3833 - accuracy: 0.8212
    Epoch 82/100
    176/176 [==============================] - 0s 842us/step - loss: 0.3833 - accuracy: 0.8224
    Epoch 83/100
    176/176 [==============================] - 0s 832us/step - loss: 0.3779 - accuracy: 0.8254
    Epoch 84/100
    176/176 [==============================] - 0s 686us/step - loss: 0.3834 - accuracy: 0.8215
    Epoch 85/100
    176/176 [==============================] - 0s 695us/step - loss: 0.3858 - accuracy: 0.8192
    Epoch 86/100
    176/176 [==============================] - 0s 667us/step - loss: 0.3801 - accuracy: 0.8304
    Epoch 87/100
    176/176 [==============================] - 0s 609us/step - loss: 0.3791 - accuracy: 0.8215
    Epoch 88/100
    176/176 [==============================] - 0s 603us/step - loss: 0.3795 - accuracy: 0.8263
    Epoch 89/100
    176/176 [==============================] - 0s 612us/step - loss: 0.3805 - accuracy: 0.8245
    Epoch 90/100
    176/176 [==============================] - 0s 611us/step - loss: 0.3808 - accuracy: 0.8162
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
    Epoch 11/100
    94/94 [==============================] - 0s 758us/step - loss: 0.4710 - accuracy: 0.7773
    Epoch 12/100
    94/94 [==============================] - 0s 768us/step - loss: 0.4677 - accuracy: 0.7766
    Epoch 13/100
    94/94 [==============================] - 0s 2ms/step - loss: 0.4673 - accuracy: 0.7759
    Epoch 14/100
    94/94 [==============================] - 0s 1ms/step - loss: 0.4651 - accuracy: 0.7783
    Epoch 15/100
    94/94 [==============================] - 0s 1ms/step - loss: 0.4623 - accuracy: 0.7803
    Epoch 16/100
    94/94 [==============================] - 0s 838us/step - loss: 0.4618 - accuracy: 0.7773
    Epoch 17/100
    94/94 [==============================] - 0s 808us/step - loss: 0.4617 - accuracy: 0.7836
    Epoch 18/100
    94/94 [==============================] - 0s 1ms/step - loss: 0.4582 - accuracy: 0.7829
    Epoch 19/100
    94/94 [==============================] - 0s 1ms/step - loss: 0.4569 - accuracy: 0.7870
    Epoch 20/100
    94/94 [==============================] - 0s 868us/step - loss: 0.4545 - accuracy: 0.7836
    Epoch 21/100
    94/94 [==============================] - 0s 1ms/step - loss: 0.4533 - accuracy: 0.7860
    Epoch 22/100
    94/94 [==============================] - 0s 1ms/step - loss: 0.4514 - accuracy: 0.7896
    Epoch 23/100
    94/94 [==============================] - 0s 1ms/step - loss: 0.4498 - accuracy: 0.7913
    Epoch 24/100
    94/94 [==============================] - 0s 1ms/step - loss: 0.4487 - accuracy: 0.7940
    Epoch 25/100
    94/94 [==============================] - 0s 828us/step - loss: 0.4479 - accuracy: 0.7930
    Epoch 26/100
    94/94 [==============================] - 0s 977us/step - loss: 0.4460 - accuracy: 0.7910
    Epoch 27/100
    94/94 [==============================] - 0s 784us/step - loss: 0.4457 - accuracy: 0.7930
    Epoch 28/100
    94/94 [==============================] - 0s 800us/step - loss: 0.4443 - accuracy: 0.7943
    Epoch 29/100
    94/94 [==============================] - 0s 830us/step - loss: 0.4431 - accuracy: 0.7963
    Epoch 30/100
    94/94 [==============================] - 0s 791us/step - loss: 0.4415 - accuracy: 0.7957
    Epoch 31/100
    94/94 [==============================] - 0s 793us/step - loss: 0.4393 - accuracy: 0.7963
    Epoch 32/100
    94/94 [==============================] - 0s 729us/step - loss: 0.4376 - accuracy: 0.8013
    Epoch 33/100
    94/94 [==============================] - 0s 750us/step - loss: 0.4359 - accuracy: 0.7983
    Epoch 34/100
    94/94 [==============================] - 0s 842us/step - loss: 0.4363 - accuracy: 0.7923
    Epoch 35/100
    94/94 [==============================] - 0s 877us/step - loss: 0.4346 - accuracy: 0.8017
    Epoch 36/100
    94/94 [==============================] - 0s 817us/step - loss: 0.4322 - accuracy: 0.7987
    Epoch 37/100
    94/94 [==============================] - 0s 931us/step - loss: 0.4327 - accuracy: 0.8027
    Epoch 38/100
    94/94 [==============================] - 0s 793us/step - loss: 0.4295 - accuracy: 0.7997
    Epoch 39/100
    94/94 [==============================] - 0s 753us/step - loss: 0.4283 - accuracy: 0.8020
    Epoch 40/100
    94/94 [==============================] - 0s 770us/step - loss: 0.4274 - accuracy: 0.8033
    Epoch 41/100
    94/94 [==============================] - 0s 864us/step - loss: 0.4260 - accuracy: 0.8017
    Epoch 42/100
    94/94 [==============================] - 0s 856us/step - loss: 0.4261 - accuracy: 0.8074
    Epoch 43/100
    94/94 [==============================] - 0s 802us/step - loss: 0.4237 - accuracy: 0.8087
    Epoch 44/100
    94/94 [==============================] - 0s 804us/step - loss: 0.4227 - accuracy: 0.7993
    Epoch 45/100
    94/94 [==============================] - 0s 774us/step - loss: 0.4203 - accuracy: 0.8060
    Epoch 46/100
    94/94 [==============================] - 0s 837us/step - loss: 0.4212 - accuracy: 0.8074
    Epoch 47/100
    94/94 [==============================] - 0s 904us/step - loss: 0.4195 - accuracy: 0.8084
    Epoch 48/100
    94/94 [==============================] - 0s 849us/step - loss: 0.4176 - accuracy: 0.8094
    Epoch 49/100
    94/94 [==============================] - 0s 758us/step - loss: 0.4165 - accuracy: 0.8107
    Epoch 50/100
    94/94 [==============================] - 0s 823us/step - loss: 0.4174 - accuracy: 0.8077
    Epoch 51/100
    94/94 [==============================] - 0s 743us/step - loss: 0.4136 - accuracy: 0.8134
    Epoch 52/100
    94/94 [==============================] - 0s 843us/step - loss: 0.4148 - accuracy: 0.8144
    Epoch 53/100
    94/94 [==============================] - 0s 900us/step - loss: 0.4135 - accuracy: 0.8144
    Epoch 54/100
    94/94 [==============================] - 0s 800us/step - loss: 0.4116 - accuracy: 0.8124
    Epoch 55/100
    94/94 [==============================] - 0s 797us/step - loss: 0.4090 - accuracy: 0.8177
    Epoch 56/100
    94/94 [==============================] - 0s 893us/step - loss: 0.4110 - accuracy: 0.8187
    Epoch 57/100
    94/94 [==============================] - 0s 895us/step - loss: 0.4089 - accuracy: 0.8144
    Epoch 58/100
    94/94 [==============================] - 0s 892us/step - loss: 0.4076 - accuracy: 0.8151
    Epoch 59/100
    94/94 [==============================] - 0s 910us/step - loss: 0.4099 - accuracy: 0.8154
    Epoch 60/100
    94/94 [==============================] - 0s 1ms/step - loss: 0.4044 - accuracy: 0.8204
    Epoch 61/100
    94/94 [==============================] - 0s 931us/step - loss: 0.4026 - accuracy: 0.8194
    Epoch 62/100
    94/94 [==============================] - 0s 727us/step - loss: 0.4044 - accuracy: 0.8157
    Epoch 63/100
    94/94 [==============================] - 0s 744us/step - loss: 0.4034 - accuracy: 0.8177
    Epoch 64/100
    94/94 [==============================] - 0s 727us/step - loss: 0.4022 - accuracy: 0.8204
    Epoch 65/100
    94/94 [==============================] - 0s 759us/step - loss: 0.4014 - accuracy: 0.8194
    Epoch 66/100
    94/94 [==============================] - 0s 744us/step - loss: 0.4002 - accuracy: 0.8197
    Epoch 67/100
    94/94 [==============================] - 0s 776us/step - loss: 0.3981 - accuracy: 0.8204
    Epoch 68/100
    94/94 [==============================] - 0s 707us/step - loss: 0.3973 - accuracy: 0.8171
    Epoch 69/100
    94/94 [==============================] - 0s 776us/step - loss: 0.3967 - accuracy: 0.8177
    Epoch 70/100
    94/94 [==============================] - 0s 798us/step - loss: 0.3970 - accuracy: 0.8174
    Epoch 71/100
    94/94 [==============================] - 0s 807us/step - loss: 0.3983 - accuracy: 0.8204
    Epoch 72/100
    94/94 [==============================] - 0s 812us/step - loss: 0.3973 - accuracy: 0.8224
    Epoch 73/100
    94/94 [==============================] - 0s 752us/step - loss: 0.3937 - accuracy: 0.8197
    Epoch 74/100
    94/94 [==============================] - 0s 883us/step - loss: 0.3919 - accuracy: 0.8244
    Epoch 75/100
    94/94 [==============================] - 0s 765us/step - loss: 0.3932 - accuracy: 0.8234
    Epoch 76/100
    94/94 [==============================] - 0s 836us/step - loss: 0.3900 - accuracy: 0.8234
    Epoch 77/100
    94/94 [==============================] - 0s 751us/step - loss: 0.3890 - accuracy: 0.8237
    Epoch 78/100
    94/94 [==============================] - 0s 776us/step - loss: 0.3896 - accuracy: 0.8237
    Epoch 79/100
    94/94 [==============================] - 0s 862us/step - loss: 0.3888 - accuracy: 0.8258
    Epoch 80/100
    94/94 [==============================] - 0s 642us/step - loss: 0.3877 - accuracy: 0.8264
    Epoch 81/100
    94/94 [==============================] - 0s 654us/step - loss: 0.3856 - accuracy: 0.8264
    Epoch 82/100
    94/94 [==============================] - 0s 691us/step - loss: 0.3867 - accuracy: 0.8221
    Epoch 83/100
    94/94 [==============================] - 0s 708us/step - loss: 0.3859 - accuracy: 0.8304
    Epoch 84/100
    94/94 [==============================] - 0s 732us/step - loss: 0.3849 - accuracy: 0.8268
    Epoch 85/100
    94/94 [==============================] - 0s 727us/step - loss: 0.3843 - accuracy: 0.8274
    Epoch 86/100
    94/94 [==============================] - 0s 699us/step - loss: 0.3820 - accuracy: 0.8318
    Epoch 87/100
    94/94 [==============================] - 0s 714us/step - loss: 0.3823 - accuracy: 0.8281
    Epoch 88/100
    94/94 [==============================] - 0s 835us/step - loss: 0.3818 - accuracy: 0.8284
    Epoch 89/100
    94/94 [==============================] - 0s 854us/step - loss: 0.3800 - accuracy: 0.8244
    Epoch 90/100
    94/94 [==============================] - 0s 769us/step - loss: 0.3811 - accuracy: 0.8311
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
           [4.47303057e-04],
           [1.74296409e-01],
           [4.30803001e-02],
           [7.02690125e-01],
           [8.68397236e-01],
           [9.65484142e-01],
           [4.69100773e-01],
           [6.59387946e-01],
           [9.35799360e-01],
           [8.53783131e-01],
           [8.69271755e-02],
           [1.55554265e-01],
           [7.64794767e-01],
           [4.06174630e-01],
           [8.31016898e-03],
           [3.73745918e-01],
           [7.57412434e-01],
           [4.23133910e-01],
           [7.71952987e-01],
           [1.00314796e-01],
           [9.74593222e-01],
           [7.51814008e-01],
           [1.26952529e-02],
           [8.23615670e-01],
           [9.88930166e-01],
           [2.08509609e-05],
           [7.55804896e-01],
           [9.48675096e-01],
           [2.04900503e-02],
           [8.99712324e-01],
           [1.51101053e-02],
           [8.43624473e-01],
           [9.57615256e-01],
           [1.12219900e-01],
           [8.83901358e-01],
           [4.34894472e-01],
           [5.74911237e-02],
           [7.95506835e-01],
           [8.93033385e-01],
           [5.27992368e-01],
           [1.43240690e-02],
           [3.23121846e-02],
           [6.71714544e-01],
           [1.04427338e-02],
           [3.60631526e-01],
           [9.23605442e-01],
           [9.48507547e-01],
           [2.43086159e-01],
           [4.69467938e-01],
           [3.73504758e-01],
           [1.97921753e-01],
           [6.50587320e-01],
           [2.46604979e-01],
           [9.31656480e-01],
           [7.96692729e-01],
           [4.74903107e-01],
           [8.08255970e-02],
           [2.45457321e-01],
           [8.67951989e-01],
           [6.94725513e-01],
           [7.00514317e-02],
           [9.15506184e-01],
           [9.28435087e-01],
           [7.00160742e-01],
           [1.44026279e-02],
           [2.84953713e-02],
           [2.10077614e-01],
           [1.55389905e-02],
           [9.86070633e-01],
           [4.10145223e-01],
           [3.33280802e-01],
           [4.63953316e-01],
           [1.19487226e-01],
           [9.11972523e-01],
           [5.39821386e-03],
           [5.05076945e-01],
           [9.57553029e-01],
           [1.91129029e-01],
           [1.38913095e-01],
           [4.21862155e-01],
           [9.28402901e-01],
           [4.16672945e-01],
           [1.82976931e-01],
           [8.19087982e-01],
           [8.88881922e-01],
           [1.03853583e-01],
           [9.10460114e-01],
           [9.37042117e-01],
           [6.67550206e-01],
           [4.54719454e-01],
           [9.12402034e-01],
           [6.39460206e-01],
           [8.99101496e-01],
           [4.24033523e-01],
           [9.47555184e-01],
           [9.77312922e-02],
           [1.29635483e-01],
           [1.05543017e-01],
           [5.79854846e-03],
           [2.05469728e-01],
           [3.41781616e-01],
           [2.99579024e-01],
           [9.71385598e-01],
           [2.05740333e-03],
           [9.46365595e-01],
           [3.22646797e-02],
           [2.41892576e-01],
           [8.17076921e-01],
           [5.81133425e-01],
           [8.33974242e-01],
           [8.54754806e-01],
           [8.70053947e-01],
           [6.31820440e-01],
           [7.29633808e-01],
           [9.03755784e-01],
           [4.51594591e-03],
           [2.27287292e-01],
           [9.14589286e-01],
           [6.58796072e-01],
           [9.73253727e-01],
           [8.39447081e-02],
           [9.94184196e-01],
           [1.68440938e-02],
           [7.36995578e-01],
           [3.16116214e-03],
           [1.60055518e-01],
           [5.04209697e-01],
           [1.70398027e-01],
           [5.28483987e-02],
           [5.21300852e-01],
           [8.60240817e-01],
           [7.32081831e-02],
           [7.56962776e-01],
           [3.46487641e-01],
           [1.03369504e-01],
           [8.83185029e-01],
           [8.65600944e-01],
           [5.01533449e-01],
           [7.73470759e-01],
           [9.79507327e-01],
           [1.69866681e-01],
           [5.50971270e-01],
           [1.44251019e-01],
           [9.87850308e-01],
           [5.90420365e-02],
           [4.06496406e-01],
           [3.53370786e-01],
           [2.76920199e-03],
           [5.46764612e-01],
           [5.32306433e-02],
           [9.37007606e-01],
           [9.61683989e-02],
           [9.62941051e-01],
           [7.65854359e-01],
           [7.17296302e-02],
           [4.55873340e-01],
           [8.95338893e-01],
           [7.99301565e-02],
           [1.23115182e-02],
           [9.59415317e-01],
           [8.85705352e-01],
           [9.65252161e-01],
           [2.87094712e-03],
           [9.35628533e-01],
           [7.08180845e-01],
           [8.40087533e-01],
           [5.66645324e-01],
           [8.94663930e-01],
           [5.46826124e-01],
           [5.02580822e-01],
           [9.06184494e-01],
           [5.81740499e-01],
           [9.55524325e-01],
           [8.97170961e-01],
           [1.05446577e-03],
           [4.98487502e-01],
           [4.80472535e-01],
           [8.11898172e-01],
           [6.23036027e-01],
           [4.91812259e-01],
           [6.57991230e-01],
           [4.75967616e-01],
           [8.02518845e-01],
           [4.93344873e-01],
           [1.05593413e-01],
           [6.50345266e-01],
           [2.84970939e-01],
           [2.40214795e-01],
           [9.58584547e-02],
           [6.61516964e-01],
           [1.26595229e-01],
           [5.98245859e-03],
           [9.58596945e-01],
           [1.02128953e-01],
           [8.09419155e-03],
           [1.11846387e-01],
           [9.94419932e-01],
           [9.42877650e-01],
           [9.14227962e-03],
           [9.05143738e-01],
           [2.20366716e-02],
           [7.52551198e-01],
           [9.08899903e-02],
           [4.28257257e-01],
           [6.63258553e-01],
           [8.59573185e-02],
           [7.25100636e-01],
           [1.31044567e-01],
           [7.35535681e-01],
           [5.26994407e-01],
           [4.77348894e-01],
           [5.82275867e-01],
           [1.27067685e-01],
           [7.88511217e-01],
           [5.88168859e-01],
           [9.42701757e-01],
           [3.60117793e-01],
           [7.56847978e-01],
           [8.56472433e-01],
           [4.84937340e-01],
           [1.33114606e-01],
           [2.18123198e-03],
           [1.09440684e-02],
           [7.30112195e-03],
           [8.79072309e-01],
           [8.96389484e-01],
           [8.83293688e-01],
           [4.20872808e-01],
           [1.77545190e-01],
           [9.34015334e-01],
           [6.18882895e-01],
           [4.44179893e-01],
           [2.65449286e-04],
           [8.18596780e-01],
           [9.09210443e-02],
           [3.95601779e-01],
           [1.60842538e-02],
           [2.58866251e-02],
           [1.28280371e-01],
           [2.81697512e-03],
           [9.84949470e-01],
           [7.60413349e-01],
           [9.70733166e-01],
           [8.92342925e-01],
           [7.48671532e-01],
           [8.47444654e-01],
           [4.64357018e-01],
           [8.88141513e-01],
           [1.91325039e-01],
           [4.37523961e-01],
           [1.52449608e-02],
           [8.45116973e-01],
           [1.16853714e-02],
           [8.80128741e-02],
           [6.84073567e-03],
           [4.90385294e-03],
           [5.56215048e-01],
           [4.46329027e-01],
           [9.03339505e-01],
           [1.38670206e-04],
           [6.48314834e-01],
           [1.35660172e-03],
           [9.82756138e-01],
           [9.32929277e-01],
           [4.40941870e-01],
           [9.23341513e-03],
           [8.24100375e-01],
           [7.62494087e-01],
           [5.36966026e-02],
           [8.46003473e-01],
           [3.64927053e-02],
           [1.78338885e-02],
           [9.84885573e-01],
           [5.69602668e-01],
           [3.76347870e-01],
           [4.34753001e-01],
           [2.41516531e-02],
           [8.91010821e-01],
           [8.10328245e-01],
           [9.09026980e-01],
           [1.80372596e-03],
           [5.21157503e-01],
           [6.09219313e-01],
           [8.74625385e-01],
           [6.91681206e-01],
           [8.84938240e-01],
           [5.17532229e-03],
           [1.62733197e-02],
           [2.00497776e-01],
           [8.65840435e-01],
           [1.26593709e-02],
           [8.83589864e-01],
           [5.89758158e-04],
           [6.37430012e-01],
           [2.24966079e-01],
           [2.93368101e-02],
           [8.07637632e-01],
           [2.62093544e-03],
           [5.81459939e-01],
           [5.22411644e-01],
           [3.29080284e-01],
           [9.79672194e-01],
           [5.61009526e-01],
           [9.59207952e-01],
           [2.37788796e-01],
           [1.04854941e-01],
           [5.21945715e-01],
           [5.69105148e-04],
           [5.25332451e-01],
           [9.05819237e-02],
           [1.79875910e-01],
           [3.72443557e-01],
           [9.42534208e-01],
           [4.93103206e-01],
           [9.82373595e-01],
           [2.85601616e-03],
           [8.05069804e-02],
           [9.77626085e-01],
           [9.06829298e-01],
           [1.94557905e-02],
           [1.46673143e-01],
           [8.36153507e-01],
           [6.39516234e-01],
           [6.66064382e-01],
           [8.13475132e-01],
           [8.74322891e-01],
           [1.23195708e-01],
           [8.63572240e-01],
           [8.55916381e-01],
           [7.77473211e-01],
           [2.05913275e-01],
           [2.11568356e-01],
           [5.43600619e-02],
           [8.04832876e-02],
           [8.50177884e-01],
           [1.30359232e-02],
           [9.11366642e-01],
           [5.97321630e-01],
           [6.84378028e-01],
           [4.62200642e-02],
           [1.28906965e-02],
           [3.94135118e-02],
           [9.91821289e-01],
           [2.57240236e-02],
           [9.59172428e-01],
           [8.73780251e-01],
           [9.78617430e-01],
           [9.16156650e-01],
           [8.85376275e-01],
           [9.63883877e-01],
           [8.09836388e-03],
           [9.89586711e-01],
           [9.12328124e-01],
           [4.09870744e-02],
           [3.26696038e-02],
           [8.62848520e-01],
           [2.52047718e-01],
           [8.77957880e-01],
           [4.12963122e-01],
           [9.02664125e-01],
           [8.74146700e-01],
           [9.57372546e-01],
           [8.56226683e-01],
           [8.84405851e-01],
           [9.48246121e-01],
           [1.34823024e-02],
           [2.90459096e-02],
           [9.81795430e-01],
           [1.47233099e-01],
           [5.67753017e-02],
           [2.01374650e-01],
           [2.96992809e-01],
           [1.29184544e-01],
           [8.77587080e-01],
           [8.77614498e-01],
           [8.83463025e-03],
           [4.66491014e-01],
           [8.88181806e-01],
           [7.77215362e-01],
           [4.40729409e-01],
           [8.72723162e-01],
           [4.17158604e-02],
           [8.46918106e-01],
           [5.48428297e-03],
           [5.20646334e-01],
           [5.42807281e-02],
           [3.87822390e-01],
           [7.84537554e-01],
           [9.42327738e-01],
           [3.76899064e-01],
           [8.55058551e-01],
           [4.26950514e-01],
           [8.83384407e-01],
           [8.93364549e-01],
           [8.65980983e-01],
           [8.14305782e-01],
           [5.01312792e-01],
           [3.59570980e-03],
           [6.31480038e-01],
           [7.59590566e-02],
           [3.62584233e-01],
           [8.29824209e-01],
           [5.74541986e-02],
           [2.71412730e-03],
           [9.18812156e-01],
           [1.60130054e-01],
           [5.32202780e-01],
           [5.41031361e-03],
           [7.57191300e-01],
           [3.10824513e-02],
           [2.01647252e-01],
           [8.73654246e-01],
           [9.50223804e-01],
           [7.26044178e-04],
           [9.93734598e-03],
           [8.49446535e-01],
           [7.97573805e-01],
           [6.00508511e-01],
           [9.44358706e-01],
           [1.57755911e-02],
           [4.20102477e-03],
           [8.11669350e-01],
           [9.26307917e-01],
           [9.75170612e-01],
           [3.37846816e-01],
           [8.77175391e-01],
           [2.20880508e-02],
           [5.51612675e-02],
           [9.49841738e-03],
           [9.38181877e-01],
           [7.53949523e-01],
           [8.11464429e-01],
           [6.08033001e-01],
           [4.35655862e-01],
           [8.68792534e-01],
           [5.15080988e-02],
           [1.63411796e-02],
           [5.20870984e-01],
           [5.78535259e-01],
           [4.34184074e-03],
           [6.86701596e-01],
           [9.67785835e-01],
           [6.33585870e-01],
           [4.74945337e-01],
           [8.24777722e-01],
           [3.33710909e-01],
           [2.80497670e-01],
           [9.34794664e-01],
           [8.71316195e-02],
           [2.34094441e-01],
           [6.03810251e-01],
           [8.86830986e-01],
           [5.85902035e-02],
           [7.81738758e-03],
           [8.93303692e-01],
           [7.39897490e-02],
           [3.10351551e-01],
           [8.75271082e-01],
           [2.71189451e-01],
           [9.06297863e-01],
           [5.43042183e-01],
           [6.54887497e-01],
           [2.99524665e-02],
           [9.82242942e-01],
           [8.77478600e-01],
           [1.56639516e-02],
           [4.46711332e-01],
           [5.82666337e-01],
           [8.30607295e-01],
           [4.75060314e-01],
           [7.96335578e-01],
           [2.87094712e-03],
           [9.07697976e-01],
           [5.41777015e-02],
           [7.02243447e-01],
           [9.13180768e-01],
           [4.14756626e-01],
           [3.25471163e-04],
           [8.79919529e-01],
           [4.36043441e-02],
           [4.20835614e-03],
           [4.60726023e-01],
           [9.14085150e-01],
           [8.37864518e-01],
           [1.24740601e-03],
           [7.31209159e-01],
           [8.98432076e-01],
           [4.19932216e-01],
           [9.75647807e-01],
           [5.57323635e-01],
           [5.97702920e-01],
           [3.12189668e-01],
           [9.32952404e-01],
           [2.68141031e-02],
           [1.76857591e-01],
           [1.71020031e-02],
           [2.21377343e-01],
           [9.72163081e-02],
           [1.59960836e-01],
           [5.75869262e-01],
           [9.60512161e-02],
           [5.11666238e-01],
           [4.72877622e-02],
           [1.84476376e-04],
           [4.29693460e-01],
           [9.76203859e-01],
           [2.74044335e-01],
           [3.16036642e-02],
           [9.37440276e-01],
           [3.41712654e-01],
           [9.94073868e-01],
           [6.52253926e-01],
           [7.38403916e-01],
           [6.88249171e-01],
           [8.75668764e-01],
           [8.74025822e-01],
           [7.33347893e-01],
           [8.47923696e-01],
           [8.18958640e-01],
           [8.59395385e-01],
           [9.44415331e-01],
           [8.03726733e-01],
           [9.30193424e-01],
           [2.54542112e-01],
           [9.74369049e-03],
           [1.35738045e-01],
           [2.75772810e-03],
           [9.27625060e-01],
           [4.35095578e-01],
           [1.56929731e-01],
           [2.90512085e-01],
           [7.28695393e-01],
           [5.32586575e-02],
           [5.69015741e-03],
           [7.32638955e-01],
           [5.89459598e-01],
           [9.94991243e-01],
           [9.59360421e-01],
           [1.62699103e-01],
           [7.96857178e-01],
           [4.13305759e-01],
           [5.69463670e-01],
           [2.22439706e-01],
           [4.70674336e-01],
           [6.98151648e-01],
           [1.13172472e-01],
           [8.05325985e-01],
           [8.46054316e-01],
           [9.51711774e-01],
           [2.82984197e-01],
           [1.26335561e-01],
           [2.47014999e-01],
           [2.57849693e-02],
           [8.76783311e-01],
           [8.84722233e-01],
           [1.34798884e-03],
           [1.25228167e-02],
           [3.47719073e-01],
           [3.54449719e-01],
           [4.12538946e-01],
           [9.33068991e-03],
           [8.52012694e-01],
           [1.62295401e-02],
           [8.67273510e-01],
           [3.77264112e-01],
           [9.54039812e-01],
           [8.82186532e-01],
           [8.88758004e-02],
           [7.14368582e-01],
           [1.75329119e-01],
           [8.77504826e-01],
           [9.48019743e-01],
           [9.68822837e-01],
           [2.31220603e-01],
           [5.58614731e-03],
           [7.39968956e-01],
           [1.47409588e-01],
           [8.12029362e-01],
           [8.36697221e-03],
           [7.45474577e-01],
           [5.75041175e-02],
           [4.61313039e-01],
           [3.37219477e-01],
           [1.90303057e-01],
           [5.40211976e-01],
           [7.64512062e-01],
           [7.33237207e-01],
           [3.88880849e-01],
           [1.22748107e-01],
           [7.04002976e-01],
           [6.51244164e-01],
           [7.57269919e-01],
           [6.39750302e-01],
           [7.92469621e-01],
           [8.99745882e-01],
           [7.87778258e-01],
           [2.39757895e-02],
           [8.88013840e-01],
           [7.82394290e-01],
           [1.51094496e-02],
           [7.48709381e-01],
           [2.61101127e-03],
           [1.92091703e-01],
           [4.41568017e-01],
           [8.28458905e-01],
           [8.86314094e-01],
           [5.65098822e-02],
           [4.59105492e-01],
           [9.75546896e-01],
           [8.55633259e-01],
           [1.74342424e-01],
           [8.54503691e-01],
           [6.99634552e-02],
           [5.58961034e-01],
           [9.37338293e-01],
           [3.43048275e-01],
           [5.40011525e-02],
           [1.24580413e-01],
           [7.97195256e-01],
           [7.38814473e-02],
           [9.14679646e-01],
           [8.82567406e-01],
           [1.03677422e-01],
           [8.30151200e-01],
           [6.96954966e-01],
           [8.11070919e-01],
           [4.02736545e-01],
           [9.89998758e-01],
           [8.46970081e-01],
           [1.25555396e-02],
           [5.93224347e-01],
           [4.60364014e-01],
           [7.54358590e-01],
           [3.81916493e-01],
           [2.16125578e-01],
           [1.87984705e-01],
           [7.13539124e-03],
           [2.81478345e-01],
           [6.90253556e-01],
           [3.50078940e-03],
           [2.06606656e-01],
           [4.55004305e-01],
           [3.73078048e-01],
           [9.15210485e-01],
           [5.11971414e-02],
           [8.16171050e-01],
           [5.85698128e-01],
           [4.12746072e-02],
           [3.76093388e-02],
           [3.59401107e-03],
           [1.86833411e-01],
           [6.16901517e-01],
           [2.15531588e-02],
           [8.77778769e-01],
           [7.83677459e-01],
           [8.55064392e-01],
           [2.96319723e-02],
           [8.69025350e-01],
           [9.93913889e-01],
           [3.41254234e-01],
           [8.97304535e-01],
           [9.42166984e-01],
           [8.42139602e-01],
           [5.21372557e-01],
           [5.48627973e-03],
           [3.87866795e-01],
           [1.45677418e-01],
           [3.26705277e-02],
           [9.70831513e-03],
           [6.63929224e-01],
           [1.47563219e-03],
           [6.05553925e-01],
           [2.01285958e-01],
           [5.88043272e-01],
           [4.45364624e-01],
           [8.11526239e-01],
           [4.57793772e-02],
           [8.36652517e-01],
           [4.35529828e-01],
           [8.86466503e-01],
           [9.39956129e-01],
           [2.05923021e-02],
           [4.50665981e-01],
           [4.15570110e-01],
           [2.70079106e-01],
           [1.97160482e-01],
           [1.81023479e-02],
           [5.53196490e-01],
           [8.91903758e-01],
           [5.21444261e-01],
           [7.81683087e-01],
           [8.52560520e-01],
           [4.04253483e-01],
           [6.24435246e-02],
           [3.18725526e-01],
           [4.50981975e-01],
           [1.24058485e-01],
           [7.72848368e-01],
           [3.50016356e-01],
           [4.98384148e-01],
           [2.14420199e-01],
           [4.78098691e-01],
           [1.40770972e-02],
           [9.10816908e-01],
           [1.20508879e-01],
           [8.70494723e-01],
           [5.67601562e-01],
           [3.60773146e-01],
           [8.86138916e-01],
           [7.79934525e-01],
           [6.86185062e-01],
           [1.74941868e-01],
           [1.65052444e-01],
           [3.43961358e-01],
           [9.07405257e-01],
           [1.59152150e-02],
           [1.05987102e-01],
           [8.42646480e-01],
           [1.59709215e-01],
           [4.98181880e-02],
           [2.06427276e-02],
           [3.50389421e-01],
           [9.18520689e-01],
           [8.95771623e-01],
           [8.39304209e-01],
           [2.76312232e-03],
           [1.33225322e-01],
           [4.75340694e-01],
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
    Epoch 11/100
    259/259 [==============================] - 0s 683us/step - loss: 0.4560 - accuracy: 0.7835
    Epoch 12/100
    259/259 [==============================] - 0s 606us/step - loss: 0.4539 - accuracy: 0.7864
    Epoch 13/100
    259/259 [==============================] - 0s 690us/step - loss: 0.4521 - accuracy: 0.7847
    Epoch 14/100
    259/259 [==============================] - 0s 834us/step - loss: 0.4502 - accuracy: 0.7851
    Epoch 15/100
    259/259 [==============================] - 0s 853us/step - loss: 0.4488 - accuracy: 0.7875
    Epoch 16/100
    259/259 [==============================] - 0s 783us/step - loss: 0.4462 - accuracy: 0.7873
    Epoch 17/100
    259/259 [==============================] - 0s 798us/step - loss: 0.4443 - accuracy: 0.7868
    Epoch 18/100
    259/259 [==============================] - 0s 2ms/step - loss: 0.4421 - accuracy: 0.7887
    Epoch 19/100
    259/259 [==============================] - 0s 1ms/step - loss: 0.4406 - accuracy: 0.7929
    Epoch 20/100
    259/259 [==============================] - 0s 824us/step - loss: 0.4387 - accuracy: 0.7930
    Epoch 21/100
    259/259 [==============================] - 0s 710us/step - loss: 0.4369 - accuracy: 0.7931
    Epoch 22/100
    259/259 [==============================] - 0s 647us/step - loss: 0.4363 - accuracy: 0.7943
    Epoch 23/100
    259/259 [==============================] - 0s 745us/step - loss: 0.4340 - accuracy: 0.7954
    Epoch 24/100
    259/259 [==============================] - 0s 659us/step - loss: 0.4326 - accuracy: 0.7987
    Epoch 25/100
    259/259 [==============================] - 0s 650us/step - loss: 0.4327 - accuracy: 0.7982
    Epoch 26/100
    259/259 [==============================] - 0s 646us/step - loss: 0.4299 - accuracy: 0.7970
    Epoch 27/100
    259/259 [==============================] - 0s 823us/step - loss: 0.4285 - accuracy: 0.7999
    Epoch 28/100
    259/259 [==============================] - 0s 853us/step - loss: 0.4269 - accuracy: 0.8011
    Epoch 29/100
    259/259 [==============================] - 0s 722us/step - loss: 0.4245 - accuracy: 0.8048
    Epoch 30/100
    259/259 [==============================] - 0s 678us/step - loss: 0.4244 - accuracy: 0.8033
    Epoch 31/100
    259/259 [==============================] - 0s 705us/step - loss: 0.4215 - accuracy: 0.8045
    Epoch 32/100
    259/259 [==============================] - 0s 655us/step - loss: 0.4205 - accuracy: 0.8054
    Epoch 33/100
    259/259 [==============================] - 0s 639us/step - loss: 0.4176 - accuracy: 0.8076
    Epoch 34/100
    259/259 [==============================] - 0s 726us/step - loss: 0.4171 - accuracy: 0.8081
    Epoch 35/100
    259/259 [==============================] - 0s 842us/step - loss: 0.4154 - accuracy: 0.8087
    Epoch 36/100
    259/259 [==============================] - 0s 1ms/step - loss: 0.4129 - accuracy: 0.8088
    Epoch 37/100
    259/259 [==============================] - 0s 861us/step - loss: 0.4119 - accuracy: 0.8102
    Epoch 38/100
    259/259 [==============================] - 0s 767us/step - loss: 0.4101 - accuracy: 0.8097
    Epoch 39/100
    259/259 [==============================] - 0s 678us/step - loss: 0.4093 - accuracy: 0.8111
    Epoch 40/100
    259/259 [==============================] - 0s 604us/step - loss: 0.4078 - accuracy: 0.8122
    Epoch 41/100
    259/259 [==============================] - 0s 595us/step - loss: 0.4054 - accuracy: 0.8136
    Epoch 42/100
    259/259 [==============================] - 0s 628us/step - loss: 0.4046 - accuracy: 0.8104
    Epoch 43/100
    259/259 [==============================] - 0s 1ms/step - loss: 0.4028 - accuracy: 0.8153
    Epoch 44/100
    259/259 [==============================] - 0s 1ms/step - loss: 0.4011 - accuracy: 0.8151
    Epoch 45/100
    259/259 [==============================] - 0s 741us/step - loss: 0.4008 - accuracy: 0.8146
    Epoch 46/100
    259/259 [==============================] - 0s 679us/step - loss: 0.3993 - accuracy: 0.8159
    Epoch 47/100
    259/259 [==============================] - 0s 663us/step - loss: 0.3977 - accuracy: 0.8166
    Epoch 48/100
    259/259 [==============================] - 0s 710us/step - loss: 0.3980 - accuracy: 0.8159
    Epoch 49/100
    259/259 [==============================] - 0s 754us/step - loss: 0.3946 - accuracy: 0.8182
    Epoch 50/100
    259/259 [==============================] - 0s 728us/step - loss: 0.3934 - accuracy: 0.8197
    Epoch 51/100
    259/259 [==============================] - 0s 680us/step - loss: 0.3924 - accuracy: 0.8211
    Epoch 52/100
    259/259 [==============================] - 0s 850us/step - loss: 0.3922 - accuracy: 0.8176
    Epoch 53/100
    259/259 [==============================] - 0s 626us/step - loss: 0.3905 - accuracy: 0.8179
    Epoch 54/100
    259/259 [==============================] - 0s 703us/step - loss: 0.3877 - accuracy: 0.8205
    Epoch 55/100
    259/259 [==============================] - 0s 765us/step - loss: 0.3898 - accuracy: 0.8212
    Epoch 56/100
    259/259 [==============================] - 0s 814us/step - loss: 0.3873 - accuracy: 0.8219
    Epoch 57/100
    259/259 [==============================] - 0s 831us/step - loss: 0.3860 - accuracy: 0.8220
    Epoch 58/100
    259/259 [==============================] - 0s 833us/step - loss: 0.3854 - accuracy: 0.8230
    Epoch 59/100
    259/259 [==============================] - 0s 667us/step - loss: 0.3851 - accuracy: 0.8237
    Epoch 60/100
    259/259 [==============================] - 0s 638us/step - loss: 0.3825 - accuracy: 0.8257
    Epoch 61/100
    259/259 [==============================] - 0s 709us/step - loss: 0.3826 - accuracy: 0.8260
    Epoch 62/100
    259/259 [==============================] - 0s 723us/step - loss: 0.3818 - accuracy: 0.8229
    Epoch 63/100
    259/259 [==============================] - 0s 950us/step - loss: 0.3805 - accuracy: 0.8262
    Epoch 64/100
    259/259 [==============================] - 0s 834us/step - loss: 0.3810 - accuracy: 0.8245
    Epoch 65/100
    259/259 [==============================] - 0s 860us/step - loss: 0.3792 - accuracy: 0.8287
    Epoch 66/100
    259/259 [==============================] - 0s 1ms/step - loss: 0.3795 - accuracy: 0.8291
    Epoch 67/100
    259/259 [==============================] - 0s 1ms/step - loss: 0.3773 - accuracy: 0.8283
    Epoch 68/100
    259/259 [==============================] - 0s 1ms/step - loss: 0.3758 - accuracy: 0.8292
    Epoch 69/100
    259/259 [==============================] - 0s 1ms/step - loss: 0.3762 - accuracy: 0.8278
    Epoch 70/100
    259/259 [==============================] - 0s 1ms/step - loss: 0.3740 - accuracy: 0.8297
    Epoch 71/100
    259/259 [==============================] - 0s 948us/step - loss: 0.3738 - accuracy: 0.8292
    Epoch 72/100
    259/259 [==============================] - 0s 692us/step - loss: 0.3738 - accuracy: 0.8291
    Epoch 73/100
    259/259 [==============================] - 0s 627us/step - loss: 0.3727 - accuracy: 0.8303
    Epoch 74/100
    259/259 [==============================] - 0s 731us/step - loss: 0.3713 - accuracy: 0.8316
    Epoch 75/100
    259/259 [==============================] - 0s 854us/step - loss: 0.3708 - accuracy: 0.8324
    Epoch 76/100
    259/259 [==============================] - 0s 868us/step - loss: 0.3712 - accuracy: 0.8326
    Epoch 77/100
    259/259 [==============================] - 0s 834us/step - loss: 0.3684 - accuracy: 0.8349
    Epoch 78/100
    259/259 [==============================] - 0s 874us/step - loss: 0.3684 - accuracy: 0.8331
    Epoch 79/100
    259/259 [==============================] - 0s 972us/step - loss: 0.3669 - accuracy: 0.8339
    Epoch 80/100
    259/259 [==============================] - 0s 930us/step - loss: 0.3674 - accuracy: 0.8324
    Epoch 81/100
    259/259 [==============================] - 0s 1ms/step - loss: 0.3644 - accuracy: 0.8378
    Epoch 82/100
    259/259 [==============================] - 0s 701us/step - loss: 0.3653 - accuracy: 0.8344
    Epoch 83/100
    259/259 [==============================] - 0s 717us/step - loss: 0.3635 - accuracy: 0.8362
    Epoch 84/100
    259/259 [==============================] - 0s 995us/step - loss: 0.3629 - accuracy: 0.8363
    Epoch 85/100
    259/259 [==============================] - 0s 852us/step - loss: 0.3623 - accuracy: 0.8361
    Epoch 86/100
    259/259 [==============================] - 0s 711us/step - loss: 0.3615 - accuracy: 0.8398
    Epoch 87/100
    259/259 [==============================] - 0s 683us/step - loss: 0.3602 - accuracy: 0.8393
    Epoch 88/100
    259/259 [==============================] - 0s 653us/step - loss: 0.3594 - accuracy: 0.8423
    Epoch 89/100
    259/259 [==============================] - 0s 673us/step - loss: 0.3602 - accuracy: 0.8367
    Epoch 90/100
    259/259 [==============================] - 0s 671us/step - loss: 0.3580 - accuracy: 0.8404
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
    Epoch 11/100
    259/259 [==============================] - 0s 597us/step - loss: 0.4259 - accuracy: 0.8062
    Epoch 12/100
    259/259 [==============================] - 0s 593us/step - loss: 0.4226 - accuracy: 0.8064
    Epoch 13/100
    259/259 [==============================] - 0s 604us/step - loss: 0.4191 - accuracy: 0.8092
    Epoch 14/100
    259/259 [==============================] - 0s 2ms/step - loss: 0.4174 - accuracy: 0.8111
    Epoch 15/100
    259/259 [==============================] - 0s 663us/step - loss: 0.4134 - accuracy: 0.8137
    Epoch 16/100
    259/259 [==============================] - 0s 600us/step - loss: 0.4121 - accuracy: 0.8163
    Epoch 17/100
    259/259 [==============================] - 0s 599us/step - loss: 0.4093 - accuracy: 0.8142
    Epoch 18/100
    259/259 [==============================] - 0s 597us/step - loss: 0.4084 - accuracy: 0.8189
    Epoch 19/100
    259/259 [==============================] - 0s 609us/step - loss: 0.4067 - accuracy: 0.8183
    Epoch 20/100
    259/259 [==============================] - 0s 632us/step - loss: 0.4022 - accuracy: 0.8195
    Epoch 21/100
    259/259 [==============================] - 0s 613us/step - loss: 0.4009 - accuracy: 0.8165
    Epoch 22/100
    259/259 [==============================] - 0s 619us/step - loss: 0.3982 - accuracy: 0.8248
    Epoch 23/100
    259/259 [==============================] - 0s 620us/step - loss: 0.3965 - accuracy: 0.8215
    Epoch 24/100
    259/259 [==============================] - 0s 621us/step - loss: 0.3959 - accuracy: 0.8255
    Epoch 25/100
    259/259 [==============================] - 0s 2ms/step - loss: 0.3933 - accuracy: 0.8263
    Epoch 26/100
    259/259 [==============================] - 0s 639us/step - loss: 0.3936 - accuracy: 0.8241
    Epoch 27/100
    259/259 [==============================] - 0s 643us/step - loss: 0.3900 - accuracy: 0.8280
    Epoch 28/100
    259/259 [==============================] - 0s 636us/step - loss: 0.3894 - accuracy: 0.8295
    Epoch 29/100
    259/259 [==============================] - 0s 649us/step - loss: 0.3891 - accuracy: 0.8265
    Epoch 30/100
    259/259 [==============================] - 0s 759us/step - loss: 0.3868 - accuracy: 0.8266
    Epoch 31/100
    259/259 [==============================] - 0s 683us/step - loss: 0.3842 - accuracy: 0.8323
    Epoch 32/100
    259/259 [==============================] - 0s 707us/step - loss: 0.3825 - accuracy: 0.8323
    Epoch 33/100
    259/259 [==============================] - 0s 881us/step - loss: 0.3815 - accuracy: 0.8305
    Epoch 34/100
    259/259 [==============================] - 0s 788us/step - loss: 0.3809 - accuracy: 0.8306
    Epoch 35/100
    259/259 [==============================] - 0s 740us/step - loss: 0.3798 - accuracy: 0.8350
    Epoch 36/100
    259/259 [==============================] - 0s 692us/step - loss: 0.3769 - accuracy: 0.8378
    Epoch 37/100
    259/259 [==============================] - 0s 943us/step - loss: 0.3768 - accuracy: 0.8327
    Epoch 38/100
    259/259 [==============================] - 0s 909us/step - loss: 0.3760 - accuracy: 0.8335
    Epoch 39/100
    259/259 [==============================] - 0s 766us/step - loss: 0.3771 - accuracy: 0.8341
    Epoch 40/100
    259/259 [==============================] - 0s 962us/step - loss: 0.3765 - accuracy: 0.8366
    Epoch 41/100
    259/259 [==============================] - 0s 813us/step - loss: 0.3738 - accuracy: 0.8354
    Epoch 42/100
    259/259 [==============================] - 0s 776us/step - loss: 0.3737 - accuracy: 0.8370
    Epoch 43/100
    259/259 [==============================] - 0s 876us/step - loss: 0.3731 - accuracy: 0.8335
    Epoch 44/100
    259/259 [==============================] - 0s 894us/step - loss: 0.3722 - accuracy: 0.8374
    Epoch 45/100
    259/259 [==============================] - 0s 969us/step - loss: 0.3693 - accuracy: 0.8374
    Epoch 46/100
    259/259 [==============================] - 0s 1ms/step - loss: 0.3700 - accuracy: 0.8338
    Epoch 47/100
    259/259 [==============================] - 0s 1ms/step - loss: 0.3681 - accuracy: 0.8385
    Epoch 48/100
    259/259 [==============================] - 0s 915us/step - loss: 0.3673 - accuracy: 0.8401
    Epoch 49/100
    259/259 [==============================] - 0s 933us/step - loss: 0.3657 - accuracy: 0.8414
    Epoch 50/100
    259/259 [==============================] - 0s 845us/step - loss: 0.3655 - accuracy: 0.8403
    Epoch 51/100
    259/259 [==============================] - 0s 940us/step - loss: 0.3645 - accuracy: 0.8431
    Epoch 52/100
    259/259 [==============================] - 1s 2ms/step - loss: 0.3657 - accuracy: 0.8396
    Epoch 53/100
    259/259 [==============================] - 0s 986us/step - loss: 0.3645 - accuracy: 0.8390
    Epoch 54/100
    259/259 [==============================] - 0s 974us/step - loss: 0.3639 - accuracy: 0.8418
    Epoch 55/100
    259/259 [==============================] - 0s 908us/step - loss: 0.3612 - accuracy: 0.8419
    Epoch 56/100
    259/259 [==============================] - 0s 844us/step - loss: 0.3614 - accuracy: 0.8425
    Epoch 57/100
    259/259 [==============================] - 0s 772us/step - loss: 0.3598 - accuracy: 0.8427
    Epoch 58/100
    259/259 [==============================] - 0s 881us/step - loss: 0.3584 - accuracy: 0.8433
    Epoch 59/100
    259/259 [==============================] - 0s 960us/step - loss: 0.3604 - accuracy: 0.8427
    Epoch 60/100
    259/259 [==============================] - 0s 990us/step - loss: 0.3581 - accuracy: 0.8441
    Epoch 61/100
    259/259 [==============================] - 0s 890us/step - loss: 0.3584 - accuracy: 0.8425
    Epoch 62/100
    259/259 [==============================] - 0s 768us/step - loss: 0.3582 - accuracy: 0.8420
    Epoch 63/100
    259/259 [==============================] - 0s 734us/step - loss: 0.3619 - accuracy: 0.8398
    Epoch 64/100
    259/259 [==============================] - 0s 912us/step - loss: 0.3545 - accuracy: 0.8461
    Epoch 65/100
    259/259 [==============================] - 0s 804us/step - loss: 0.3555 - accuracy: 0.8473
    Epoch 66/100
    259/259 [==============================] - 0s 753us/step - loss: 0.3540 - accuracy: 0.8437
    Epoch 67/100
    259/259 [==============================] - 0s 699us/step - loss: 0.3540 - accuracy: 0.8465
    Epoch 68/100
    259/259 [==============================] - 0s 689us/step - loss: 0.3544 - accuracy: 0.8439
    Epoch 69/100
    259/259 [==============================] - 0s 650us/step - loss: 0.3527 - accuracy: 0.8462
    Epoch 70/100
    259/259 [==============================] - 0s 684us/step - loss: 0.3527 - accuracy: 0.8483
    Epoch 71/100
    259/259 [==============================] - 0s 717us/step - loss: 0.3514 - accuracy: 0.8484
    Epoch 72/100
    259/259 [==============================] - 0s 684us/step - loss: 0.3506 - accuracy: 0.8508
    Epoch 73/100
    259/259 [==============================] - 0s 660us/step - loss: 0.3534 - accuracy: 0.8472
    Epoch 74/100
    259/259 [==============================] - 0s 613us/step - loss: 0.3509 - accuracy: 0.8475
    Epoch 75/100
    259/259 [==============================] - 0s 609us/step - loss: 0.3500 - accuracy: 0.8476
    Epoch 76/100
    259/259 [==============================] - 0s 605us/step - loss: 0.3503 - accuracy: 0.8470
    Epoch 77/100
    259/259 [==============================] - 0s 783us/step - loss: 0.3473 - accuracy: 0.8487
    Epoch 78/100
    259/259 [==============================] - 0s 659us/step - loss: 0.3472 - accuracy: 0.8504
    Epoch 79/100
    259/259 [==============================] - 0s 2ms/step - loss: 0.3484 - accuracy: 0.8483
    Epoch 80/100
    259/259 [==============================] - 0s 1ms/step - loss: 0.3459 - accuracy: 0.8516
    Epoch 81/100
    259/259 [==============================] - 0s 1ms/step - loss: 0.3461 - accuracy: 0.8477
    Epoch 82/100
    259/259 [==============================] - 0s 828us/step - loss: 0.3453 - accuracy: 0.8495
    Epoch 83/100
    259/259 [==============================] - 0s 851us/step - loss: 0.3453 - accuracy: 0.8501
    Epoch 84/100
    259/259 [==============================] - 0s 735us/step - loss: 0.3441 - accuracy: 0.8502
    Epoch 85/100
    259/259 [==============================] - 0s 711us/step - loss: 0.3449 - accuracy: 0.8528
    Epoch 86/100
    259/259 [==============================] - 0s 1ms/step - loss: 0.3445 - accuracy: 0.8500
    Epoch 87/100
    259/259 [==============================] - 0s 1ms/step - loss: 0.3456 - accuracy: 0.8500
    Epoch 88/100
    259/259 [==============================] - 0s 758us/step - loss: 0.3412 - accuracy: 0.8534
    Epoch 89/100
    259/259 [==============================] - 0s 769us/step - loss: 0.3427 - accuracy: 0.8510
    Epoch 90/100
    259/259 [==============================] - 0s 747us/step - loss: 0.3430 - accuracy: 0.8501
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
    Epoch 11/100
    176/176 [==============================] - 0s 863us/step - loss: 0.1038 - accuracy: 0.8144
    Epoch 12/100
    176/176 [==============================] - 0s 874us/step - loss: 0.1033 - accuracy: 0.8167
    Epoch 13/100
    176/176 [==============================] - 0s 993us/step - loss: 0.1027 - accuracy: 0.8160
    Epoch 14/100
    176/176 [==============================] - 0s 938us/step - loss: 0.1025 - accuracy: 0.8160
    Epoch 15/100
    176/176 [==============================] - 0s 841us/step - loss: 0.1020 - accuracy: 0.8146
    Epoch 16/100
    176/176 [==============================] - 0s 728us/step - loss: 0.1019 - accuracy: 0.8165
    Epoch 17/100
    176/176 [==============================] - 0s 814us/step - loss: 0.1013 - accuracy: 0.8171
    Epoch 18/100
    176/176 [==============================] - 0s 729us/step - loss: 0.1012 - accuracy: 0.8180
    Epoch 19/100
    176/176 [==============================] - 0s 686us/step - loss: 0.1001 - accuracy: 0.8206
    Epoch 20/100
    176/176 [==============================] - 0s 678us/step - loss: 0.1002 - accuracy: 0.8194
    Epoch 21/100
    176/176 [==============================] - 0s 705us/step - loss: 0.0996 - accuracy: 0.8185
    Epoch 22/100
    176/176 [==============================] - 0s 693us/step - loss: 0.0994 - accuracy: 0.8180
    Epoch 23/100
    176/176 [==============================] - 0s 907us/step - loss: 0.0989 - accuracy: 0.8212
    Epoch 24/100
    176/176 [==============================] - 0s 853us/step - loss: 0.0985 - accuracy: 0.8194
    Epoch 25/100
    176/176 [==============================] - 0s 864us/step - loss: 0.0980 - accuracy: 0.8222
    Epoch 26/100
    176/176 [==============================] - 0s 2ms/step - loss: 0.0981 - accuracy: 0.8215
    Epoch 27/100
    176/176 [==============================] - 0s 839us/step - loss: 0.0976 - accuracy: 0.8231
    Epoch 28/100
    176/176 [==============================] - 0s 654us/step - loss: 0.0971 - accuracy: 0.8242
    Epoch 29/100
    176/176 [==============================] - 0s 662us/step - loss: 0.0969 - accuracy: 0.8249
    Epoch 30/100
    176/176 [==============================] - 0s 703us/step - loss: 0.0969 - accuracy: 0.8238
    Epoch 31/100
    176/176 [==============================] - 0s 660us/step - loss: 0.0964 - accuracy: 0.8210
    Epoch 32/100
    176/176 [==============================] - 0s 702us/step - loss: 0.0960 - accuracy: 0.8242
    Epoch 33/100
    176/176 [==============================] - 0s 700us/step - loss: 0.0959 - accuracy: 0.8263
    Epoch 34/100
    176/176 [==============================] - 0s 819us/step - loss: 0.0953 - accuracy: 0.8265
    Epoch 35/100
    176/176 [==============================] - 0s 847us/step - loss: 0.0952 - accuracy: 0.8272
    Epoch 36/100
    176/176 [==============================] - 0s 848us/step - loss: 0.0948 - accuracy: 0.8240
    Epoch 37/100
    176/176 [==============================] - 0s 937us/step - loss: 0.0949 - accuracy: 0.8242
    Epoch 38/100
    176/176 [==============================] - 0s 815us/step - loss: 0.0947 - accuracy: 0.8268
    Epoch 39/100
    176/176 [==============================] - 0s 909us/step - loss: 0.0944 - accuracy: 0.8256
    Epoch 40/100
    176/176 [==============================] - 0s 981us/step - loss: 0.0943 - accuracy: 0.8252
    Epoch 41/100
    176/176 [==============================] - 0s 738us/step - loss: 0.0939 - accuracy: 0.8281
    Epoch 42/100
    176/176 [==============================] - 0s 705us/step - loss: 0.0933 - accuracy: 0.8267
    Epoch 43/100
    176/176 [==============================] - 0s 689us/step - loss: 0.0933 - accuracy: 0.8268
    Epoch 44/100
    176/176 [==============================] - 0s 678us/step - loss: 0.0931 - accuracy: 0.8279
    Epoch 45/100
    176/176 [==============================] - 0s 687us/step - loss: 0.0929 - accuracy: 0.8252
    Epoch 46/100
    176/176 [==============================] - 0s 750us/step - loss: 0.0926 - accuracy: 0.8286
    Epoch 47/100
    176/176 [==============================] - 0s 963us/step - loss: 0.0922 - accuracy: 0.8297
    Epoch 48/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.0924 - accuracy: 0.8283
    Epoch 49/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.0923 - accuracy: 0.8265
    Epoch 50/100
    176/176 [==============================] - 0s 893us/step - loss: 0.0915 - accuracy: 0.8327
    Epoch 51/100
    176/176 [==============================] - 0s 768us/step - loss: 0.0917 - accuracy: 0.8311
    Epoch 52/100
    176/176 [==============================] - 0s 2ms/step - loss: 0.0914 - accuracy: 0.8284: 0s - loss: 0.0929 - accuracy
    Epoch 53/100
    176/176 [==============================] - 0s 711us/step - loss: 0.0909 - accuracy: 0.8293
    Epoch 54/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.0913 - accuracy: 0.8279
    Epoch 55/100
    176/176 [==============================] - 0s 742us/step - loss: 0.0910 - accuracy: 0.8284
    Epoch 56/100
    176/176 [==============================] - 0s 734us/step - loss: 0.0907 - accuracy: 0.8274
    Epoch 57/100
    176/176 [==============================] - 0s 759us/step - loss: 0.0903 - accuracy: 0.8272
    Epoch 58/100
    176/176 [==============================] - 0s 754us/step - loss: 0.0904 - accuracy: 0.8274
    Epoch 59/100
    176/176 [==============================] - 0s 787us/step - loss: 0.0899 - accuracy: 0.8265
    Epoch 60/100
    176/176 [==============================] - 0s 721us/step - loss: 0.0901 - accuracy: 0.8304
    Epoch 61/100
    176/176 [==============================] - 0s 698us/step - loss: 0.0897 - accuracy: 0.8316
    Epoch 62/100
    176/176 [==============================] - 0s 612us/step - loss: 0.0896 - accuracy: 0.8311
    Epoch 63/100
    176/176 [==============================] - 0s 659us/step - loss: 0.0890 - accuracy: 0.8332
    Epoch 64/100
    176/176 [==============================] - 0s 710us/step - loss: 0.0894 - accuracy: 0.8288
    Epoch 65/100
    176/176 [==============================] - 0s 715us/step - loss: 0.0889 - accuracy: 0.8309
    Epoch 66/100
    176/176 [==============================] - 0s 717us/step - loss: 0.0887 - accuracy: 0.8364
    Epoch 67/100
    176/176 [==============================] - 0s 801us/step - loss: 0.0885 - accuracy: 0.8331
    Epoch 68/100
    176/176 [==============================] - 0s 714us/step - loss: 0.0888 - accuracy: 0.8359
    Epoch 69/100
    176/176 [==============================] - 0s 700us/step - loss: 0.0887 - accuracy: 0.8324
    Epoch 70/100
    176/176 [==============================] - 0s 714us/step - loss: 0.0883 - accuracy: 0.8306
    Epoch 71/100
    176/176 [==============================] - 0s 721us/step - loss: 0.0879 - accuracy: 0.8343
    Epoch 72/100
    176/176 [==============================] - 0s 708us/step - loss: 0.0881 - accuracy: 0.8375
    Epoch 73/100
    176/176 [==============================] - 0s 665us/step - loss: 0.0875 - accuracy: 0.8309
    Epoch 74/100
    176/176 [==============================] - 0s 656us/step - loss: 0.0878 - accuracy: 0.8377
    Epoch 75/100
    176/176 [==============================] - 0s 719us/step - loss: 0.0876 - accuracy: 0.8348
    Epoch 76/100
    176/176 [==============================] - 0s 665us/step - loss: 0.0872 - accuracy: 0.8354
    Epoch 77/100
    176/176 [==============================] - 0s 693us/step - loss: 0.0874 - accuracy: 0.8341
    Epoch 78/100
    176/176 [==============================] - 0s 690us/step - loss: 0.0866 - accuracy: 0.8341
    Epoch 79/100
    176/176 [==============================] - 0s 722us/step - loss: 0.0872 - accuracy: 0.8329
    Epoch 80/100
    176/176 [==============================] - 0s 691us/step - loss: 0.0879 - accuracy: 0.8368
    Epoch 81/100
    176/176 [==============================] - 0s 900us/step - loss: 0.0866 - accuracy: 0.8338
    Epoch 82/100
    176/176 [==============================] - 0s 722us/step - loss: 0.0867 - accuracy: 0.8379
    Epoch 83/100
    176/176 [==============================] - 0s 766us/step - loss: 0.0864 - accuracy: 0.8380
    Epoch 84/100
    176/176 [==============================] - 0s 654us/step - loss: 0.0863 - accuracy: 0.8361
    Epoch 85/100
    176/176 [==============================] - 0s 644us/step - loss: 0.0866 - accuracy: 0.8334
    Epoch 86/100
    176/176 [==============================] - 0s 616us/step - loss: 0.0861 - accuracy: 0.8386
    Epoch 87/100
    176/176 [==============================] - 0s 681us/step - loss: 0.0862 - accuracy: 0.8382
    Epoch 88/100
    176/176 [==============================] - 0s 721us/step - loss: 0.0857 - accuracy: 0.8388
    Epoch 89/100
    176/176 [==============================] - 0s 709us/step - loss: 0.0854 - accuracy: 0.8357
    Epoch 90/100
    176/176 [==============================] - 0s 1ms/step - loss: 0.0860 - accuracy: 0.8373
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
