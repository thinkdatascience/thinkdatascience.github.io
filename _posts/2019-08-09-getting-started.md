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


