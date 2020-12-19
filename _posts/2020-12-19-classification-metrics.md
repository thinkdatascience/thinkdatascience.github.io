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

We have four different terms here. Lets understand each of them.

True Positive: You predicted positive and it’s true.

A person has a disease, and a model predicted it true.

True Negative: A model predicted negative, and it's true.

A model predicted a person doesn't have disease and, it's true.

False Positive: A model predicted Positive, but it's false.

You predicted a person has a disease but in reality, he doesn't have.

This is also known as <b> Type I</b> Error. 

False Negative: You predicted Negative, but it's false.

A model predicted that a person is not having a disease but he actually has.

False Negative is also commonly called as <b> Type II</b> Error. 




