---
title: When to use Gradient Descent or Normal Equation with Linear Regression.
author: Akshay Adlakha & Akshaykumar Rao
date: 2020-12-17 08:55:00 +0800
categories: [Machine Learning, Python]
tags: [Linear Regression]
math: true 
mermaid: true
---


## Gradient Descent

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
