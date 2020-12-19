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

For example, when we have 100 samples in our data out of which 95 are zero's and rest 5 are one's. In this case, our model will predict mostly zero and give 95% accuracy. But, in reality this model is not effective because when it has to predict one it will predict zero. So, whenever we have imbalance classes in our data Accuracy is not a good measure to evaluate. We have different other ways to assess the performance of our model. In this post, we will explore various evaluation metrics and its python implementation from scratch. 

## Confusion Matrix

Confusion matrix is a table with 4 different combinations of predicted and actual values. It is widely used to evaluate the perfomance of classification model. Based on Confusion matrix, there are different metrics can be used such as Accuracy, Precision, Recall, F-Score, Specificity. We will go through each of these metrics.

![upload-image](/assets/img/sample/confusion.png)

We have four different terms here. Lets understand each of them.

<b>True Positive</b>: You predicted positive and it’s correct.

A person has a disease, and a model predicted correctly.

<b>True Negative</b>: A model predicted negative, and it's correct.

A model predicted a person doesn't have disease and, it's correct.

<b>False Positive</b>: A model predicted Positive, but it's incorrect.

You predicted a person has a disease but in reality, he doesn't have.

This is also known as <b> Type I</b> Error. 

<b>False Negative</b>: You predicted Negative, but it's incorrect.

A model predicted that a person is not having a disease but he actually has.

False Negative is also commonly called as <b> Type II</b> Error. 

Here we have a method to create a confusion matrix in Python.

``` python
  
  def ConfusionMatrix(y_true,y_pred):
     """
     :type y_true: numpy.ndarray
     :type y_pred: numpy.ndarray
     :rtype: float
     """
     values =len(set(y_true))
     result = y_true*values + y_pred;
     result = np.histogram(result,bins=values*values)
     result = np.reshape(result[0],(values,values))
     return result

```

We can also create this using [Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html).

## Accuracy

Accuracy is number of true results to the total number of cases.

$$ Accuracy = {(TP+TN)} \over (TP+FP+FN+TN) $$ 

Now, we have Python Implementation to calculate an accuracy of our model. We will use the same method to create a confusion matrix throughout this post to calculate other metrics.

```python
  
   def Accuracy(y_true,y_pred):
     """
     :type y_true: numpy.ndarray
     :type y_pred: numpy.ndarray
     :rtype: float
    
     """
     cm = ConfusionMatrix(y_true, y_pred)
     correct_pred_sum = cm.trace()
     length_test = len(y_true)
     accuracy = correct_pred_sum / length_test
     return accuracy
```

When to use: Use Accuracy when we have balanced classes or not skewed. It should not be used with imbalanced dataset because model can be reasonably accurate, but not at all valuable.

## Precision

Precision tells us what proportions of predicted positives is truly positives.

Precision = (TP)/(TP+FP)

Python Implementation:

```python
   def Precision(y_true,y_pred):
      """
      :type y_true: numpy.ndarray
      :type y_pred: numpy.ndarray
      """
      cm = ConfusionMatrix(y_true, y_pred)
      return(np.diag(cm) / np.sum(cm, axis = 0)) 
```

When to use: Precision can be used when we want to be sure of our prediction because being precise will leave other assumptions unnoticed.

## Recall

Now we have Recall, it is very important and useful metrics. It tells us that what proportions of actual positives are predicted correctly.

Recall = (TP)/(TP+FN)

Lets see Python implementation:

```python
   def Recall(y_true,y_pred):
     """
     :type y_true: numpy.ndarray
     :type y_pred: numpy.ndarray
     """
     cm = ConfusionMatrix(y_true, y_pred)
     return(np.diag(cm) / np.sum(cm, axis = 1))
```

When to use: Recall is used when we want to predict as many positive as possible. 

Recall and Precision both are useful metric for imbalanced dataset.

## F-Score

F-Score is another useful metric in evaluating classification model. It is a trade-off between Precision and Recall.

The F-1 score is a number between 0 and 1 and is the harmonic mean of precision and recall.

F-1 score = 2* Precision * Recall / Precision + Recall

Now we see our python method to find an F-1 score.

``` python
    def Recall(y_true,y_pred):
       recall = Recall(y_true, y_pred)
       precison = Precision(y_true, y_pred)
       F1score = (2*recall*precison)/ (recall + precison)
       return F1score
```

When to use: Use F-1 score when we want to have a model with good Precison and Recall.

> If you are a Doctor and you want to detect disease, you want to be sure that the person you detect has a disease (Precision) and you also want to detect as many persons with disease (Recall) as possible. Here, The F1 score manages this tradeoff between Precision and Recall.

## AUC ROC

AUC is an Area under ROC curve. It indicates how well the probabilities from the positive classes are separated from the negative classes.

We have two more terms to understand AUC ROC.

<b>Sensitivity</b> is same as Recall. It is just the proportion of trues our model is capturing. It is also known as True Positive Rate(TPR).

Sensitivty = TPR(True Positive Rate)= Recall = TP/(TP+FN)

<b>Specificity</b> is just the proportion of falses our model is capturing. Also known as False Positive rate.

Specificity = FPR(False Positive Rate)= FP/(TN+FP)

When we plot these two TPR and FPR, we get ROC curve. 

When to use: AUC measures how well predictions are ranked, rather than their absolute values. So, for example, if you as a banker want to find customers who will respond to a new offer. AUC is a good metric to use since the predictions ranked by probability is the order in which you will create a list of customers. 

Moreover, it is threshold-invariant metric. It measures the quality of the model’s predictions irrespective of what threshold is chosen, unlike Accuracy F1 score or other metrics which depend on the choice of threshold.

## Conclusion

So, building a model is not only a task in machine learning. It is very important to evaluate our different models against each other and so, it is important to choose right evaluation metric to interpret the results. 

The following points to consider while choosing a metric:

- Always be careful of what you are predicting. Keep your business objective in mind. There might be a case where Type I error is important and somewhere Type II error is important.
- How the choice of evaluation metric might affect your final predictions.



