---
title: Bias-Variance Tradeoff
author: Akshay Adlakha & Akshaykumar Rao
date: 2020-12-14 00:34:00 +0800
categories: [Machine Learning, Python]
tags: [Machine Learning]
math: true
mermaid: true
---

The errors associated with machine learning models are of two types:
- Bias
- Variance

It is important to understand these two errors and the tradeoff to minimize the total error and avoid the Underfitting & Overfitting of the model.

![upload-image](/assets/img/sample/bises.png)

## Bias

Bias indicates how far is the model prediction from the correct prediction. It occurs due to simpler assumptions made by the model to make the predictions. The model with high bias pays very little attention to training data and oversimplifies the model. In other words, the model leads to underfitting i.e, high training, and a high testing error. 

## Variance

Variance reflects the spreadness. How much the predictions for a given point vary between different realizations of the model. The model with high variance pays much attention to training data i.e, tries to capture everything from the training data, and end up with a very complex model. When it comes to testing data, it doesn't generalize well. So, such a model leads to overfitting i.e. a low training error but high testing error. 


## Bias-Variance Tradeoff

![upload-image](/assets/img/sample/Bias1.png)

From the above figure, we can see how these two terms are important. If any of these two gets inflated or deflated, it leads to the overfitting or underfitting situation. It is necessary to balance between bias and variance. We don't want to have a model with high errors. A balance between bias and variance minimizes the total error of the model. This balance is known as the Bias-Variance tradeoff.

![upload-image](/assets/img/sample/Bias2.png)

This figure clearly shows the balance between bias and variance.
