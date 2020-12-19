---
title: Classification Evaluation metrics every Data Scientist must know And when exactly to use them?
author: Akshay Adlakha & Akshaykumar Rao
date: 2020-12-19 08:10:00 +0800
categories: [Blogging, Tutorial]
tags: [writing]
math: true
mermaid: true
---


## Classification Evaluation metrics

After fitting a classification model, you have to evaluate your model. We try to increase the accuracy of our model. But, in practice, Accuracy is actually a good measure to evalute the performance of our classification model. Yes, but it depends on the nature of problem. Sometimes it is good, sometimes it is bad.  

> Every business problem is a little different, and it should be optimized differently.

For example, when we have 100 samples in our data out of which 95 are zero's and rest 5 are one's. In this case, our model will predict mostly zero and give 95% accuracy. But, in reality this model is not effective because when it has to predict one it will predict zero. So, whenever we have imbalance classes in our data Accuracy is not a good measure to evaluate. We have different other ways to assess the performance of our model. In this post, we will explore various evaluation metrics.

# Confusion Matrix

Confusion matrix is a table with 4 different combinations of predicted and actual values. It is widely used to evaluate the perfomance of classification model. Based on Confusion matrix, there are different metrics can be used such as Accuracy, Precision, Recall, F-Score, Specificity. We will go through each of these metrics.

![upload-image](/assets/img/sample/confusion.png)





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


