---
title: Linear Regression - Assumptions and Drawbacks
author: Akshay Adlakha & Akshaykumar Rao
date: 2020-12-18 08:10:00 +0800
categories: [Blogging, Tutorial]
tags: [writing]
math: true
mermaid: true
---

## Assumptions of Linear Regression

- **Assumption 1: Linear Relationship**
> There is a linear relationship between the independent variables and the dependent variable.

This assumption constraints the model to following type:

$$ Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_kX_k + \epsilon $$                      

where, \\( \epsilon \\) is the error term.

The relationship between the independent variables and the dependent variable can be checked using scatterplot. If you relationship between the two variables is not linear, then you can try applying non-linear transformation such as log, reciprocal or square root to that independent variable and/or dependent variable.

- **Assumption 2: Independence**
> Residuals are independent.

If the residuals are not independent, then there will be some sort of patern among the residuals which the linear regression model will not be able to capture. 
To check the independence of residuals, we can detect the pattern in the plot of residuals vs predicted (or actual) values. Another way is to use the [Durbin-Watson](https://en.wikipedia.org/wiki/Durbin%E2%80%93Watson_statistic) test, which measures the degree of correlation of each residual error with the previous residual error. 

- **Assumption 3: Homoscedasticity**
> The residuals have constant variance and not a function of y (or x)

Homoscedasticity is the is the property of a dataset to have a constant variance. 
If the residuals have a variance that is a function of y (called heteroscedasticity), then they are no longer identically distributed and the regression results will be difficult to interpret.  One way to detect heterscedasticity is to create a scatterplot of fitted value vs residual. If the residuals seem to become much more spread out as the fitted values increase, then it is a sign of heterscedasticity. 
Heteroscedasticity can be treated by transforming the dependent variable e.g. log transform.


- **Assumption 4: Normality**
> The error terms are normally distributed with a mean of 0.

The error terms (or residuals) are normally ditributed and also have the same (but unknown) variance.
If the residuals are not normally distributed then their randomness is lost and the model is no longer able to explain the relationship bwtween the independednt and dependent variable.

E(y) = E( //(\beta_0 + \beta_iX_i +\epsilon (i) \\) )       

Only if E(//( \epsilon \\) (i)) = 0,  then the expectation of the target variable y is equal to the to the expectation of the model. 

To check if the residuals are normally distributed, we can visualise them using Q-Q plot. 
In the Q-Q plot or the Quantile-Quantile plot, the normality assumption is met if the points on the plot roughly form a straight line. 

Make sure that you remove the outliers as they can have a huge impact on the Q-Q plot.

- **Assumption 5: Multicollinearity**
> The independent variables are linearly independent of each other i.e. there os no multicollinearity >in the data.

To learn more about how to detect and treat multicollinearity in the dataset, refer to [this](https://thinkdatascience.github.io/posts/Multicollinearity&VIF/) link.

## Drawbacks of Linear Regression

So, we have talked about the assumptions of Linear Regression. Now, it is time to discuss some of the drawbacks of Linear Regression.

<b>Sensitive to Outliers</b>: Outliers of a data set are extreme values that deviate from the other data points of the distribution. They can affect the performance of a Machine Learning model and lead to low accuracy. 

Linear Regression model is very sensitive to Outliers. A single outlier in the data can change the best fitted line. Hence outliers must be taken care of appropriately before linear regression is applied.

<b>Can’t be used with Binary Outcomes</b>: Linear Regression model, or errors associated with it can’t be used to predict the binary variables such as True or False because as we know Linear Regression model works with continuous features and its error is unbound. But, with binary variables we expect to predict only two values. 

<b>Data Independence</b>:  One of the main assumptions of Linear Regression is No multicollinearity. But in practice, we have correlated features. They are not independent of each other. Therefore, Multicollinearity must be removed from the dataset in order to get accurate results from Linear Regression model.

<b>Overfitting</b>:  Overfitting is a situation when a model tries to capture too closely to the data that it even captures noise as well. This affects the performance of a model on the unseen data. Thus, reduces its accuracy. 

In general, Linear Regression model is most likely to overfit. But, There are different ways we can overcome an overfitting situation.

- Regularization: It is a technique that can reduce the complexity of a model which in turn reduces the risk of overfitting. 

- Gradient Descent: Gradient Descent method is an iterative alogorithm to optimize the parameters. Thus, it minimizes the error of the model.
