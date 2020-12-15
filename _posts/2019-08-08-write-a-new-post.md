---
title: Writing a New Post
author: Cotes Chung
date: 2019-08-08 14:10:00 +0800
categories: [Blogging, Tutorial]
tags: [writing]
math: true
mermaid: true
---


## R<sup>2</sup> and Adjusted R<sup>2</sup> 

After fitting a regression model, you have to determine how well the model fits to the data. Is it good enough to generalize to unseen data? or Does it do a good job of explaining changes in the dependent variable?. In this post, we explore R-Squared and Adjusted R-Squared, its sigficance and limitations. 

> R<sup>2</sup> is a goodness of fit measure. It tells us that how close the data are to the fitted line. 

R<sup>2</sup> is calculated as follows:

R<sup>2</sup> = Explained variation / Total variation

or, R<sup>2</sup> = 1 −  Unexplained Variation / Total Variation


This gives us the percentage of the response variable variation that is explained by our linear regression model.

It is also known by other names such as Coefficient of Determination, or the coefficient of multiple determination for multiple regression.

We can visualize the effect of R-Squared value. It represents how the scatterness around the regression line.

![upload-image](/assets/img/sample/Rsquared.png)

Let's say, R-Squared value for the model on the left is between 15-20% and for the model on the right is 80%. It means that how much of the variance is explained by the model. In general, We want our data points close to the regression line. However, in practice we don't get a model with 100% R-Squared value i.e all the data points fall exactly on the regression line.

## Limitations of R-Squared

R-Squared value gives an estimate of the relationship between the features and response feature. However it has some limitations.

- It doesn't inform whether a chosen model is good or bad.
- Never tells that data and predictions are biased or not.

A model can have a low or a high R-Squared value. And, it is not necessarily good or bad. A model with low R-Squared value can be a good model or a model with high R-Squared value can be a poor fitted model, and vice-versa. 


