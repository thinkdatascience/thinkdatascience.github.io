---
title: When to use Gradient Descent or the Normal Equation with Linear Regression.
author: Akshay Adlakha & Akshaykumar Rao
date: 2020-12-17 08:55:00 +0800
categories: [Machine Learning, Python]
tags: [VIF]
math: true 
mermaid: true
---

In this tutorial we will learn when to use Gradient Desent or the Normal Equation method with Linear Regression. Both of these methods help to find optimum value of theta and find the best fit line for our model, or minimize the error. 

In statistics, Linear regression is a linear approach to find the relationship between a dependent variable and one or more independent variables. Let X be the independent variable and Y be the dependent variable. So, a linear relatio between these two variables as follows:

Y = mX + C

m is the slope of the line and C is the Y intercept.

We have mean squared error, one of the loss function associated with a linear regression model. We aim to minimize the error to obtain optimum value of m and c. 

![upload-image](/assets/img/sample/mse.png)

Now we know the loss function, lets see the interesting part — minimizing it and finding m and c.

## Gradient Descent

Gradient descent is an iterative optimization algorithm to find the minimum of a function.  

We can relate the functionality of a Gradient Descent method to a person climbing down a hill. He goes down the slope and takes steps based on the position. When the slope steep, he takes large steps and small stelps when the slope is less steep. He stops when he reaches to the bottom of a hill. In similar way, Gradient Descent method works. It takes step in the direction of negative gradient and minimize the error. It uses Learning rate which decides how long it takes a step to descend. 

This is four step process.

-  Initiatize m and c, let m = 0 and c = 0. Let L be the learning rate. In general, L is chosen to be a small value for good accuracy. 
-  Then, Find the partial derivative of the loss function with respect to m, and substitute in the current values of x, y, m and c in it to find the derivative value D.

![upload-image](/assets/img/sample/gd1.png)

![upload-image](/assets/img/sample/gd.png)

- Using the following equation, we update the value of m and c.

![upload-image](/assets/img/sample/gd2.png)

- Repeat Steps 1 to 3 until we acheive the minimum error.  

## Normal Equation

Gradient descent is one of the way to minimize the cost function J. We have another way to do so, Normal equation. This performs the minimization explicitly and without sticking to an iterative algorithm. In the "Normal Equation" method, we minimize J by explicitly taking its derivatives with respect to the θ<sub>J</sub>’s, and setting them to zero. This way we can find the optimum theta value without using iteration. The normal equation formula is given below: 

θ=(X<sup>T</sup>X)<sup>−1</sup>X<sup>T</sup>y

It is also known as Closed form solution.

Feature scaling is not required with the normal equation.

Here are some comparison points between Gradient Descent and the Normal equation.

- There is a need to choose learning parameter alpha in Gradient Descent, whereas it is not required with the normal equation.
- Gradient Descent method needs many iteration to converge but there is no need to iterate with the normal equation.
- The complexity of Gradient Descent method is O(kn<sup>2</sup>), whereas the normal equation method has the complexity of O(n<sup>3</sup>) because it computes an inverse of X<sup>T</sup>X.
- The Normal Equation method comes to be slow when we have large number of features, n. In this case, Gradient Descent method works well. In practice, when n exceeds 10,000 it might be a good time to go from a normal solution to a Gradient Descent method as matrix multiplication is very expensive with the large value of n.  


## Normal Equation Noninvertibility

If X<sup>T</sup>X is noninvertible, the common causes might be having:

- There are Redundant features, where two features are very closely related (i.e. they are linearly dependent).
- Too many features (e.g. number of features is appoximately equal to number of samples. To overcome this, either remove some features or use "regularization".

Solution: Removing a feature that is linearly dependent with another or removing one or more features when there are too many features in the dataset.
