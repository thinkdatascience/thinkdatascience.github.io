---
title: Different types of Statstical Tests
author: Akshay Adlakha & Akshaykumar Rao
date: 2020-12-14 20:55:00 +0800
categories: [Machine Learning, Python]
tags: [VIF]
math: true 
mermaid: true
---


## Different types of Statstical Tests

There is an importance of understanding the relationships between variables before building predictive models, So here are some different statstics tests avialble:

### Pearson Correlation
A Pearson Correlation test defines the strength of the association between 2 continuous variables. It is also known as the “product moment correlation coefficient” (PMCC).

It ranges between -1 and +1 that indicates to which extent 2 variables are linearly related.

Formula:

$$ r =\frac{\sum\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right)}{\sqrt{\sum\left(x_{i}-\bar{x}\right)^{2} \sum\left(y_{i}-\bar{y}\right)^{2}}} $$

r	=	correlation coefficient

\\(x_{i}\\)	=	values of the x-variable in a sample

\\(\bar{x}\\)	=	mean of the values of the x-variable

\\(y_{i}\\)	=	values of the y-variable in a sample

\\(\bar{y}\\)	=	mean of the values of the y-variable


### Chi-Square Test
A Chi-Square Test, also written as χ² test, is used to find the strength of the association between two categorical variables.

Formula:

$$ \chi^{2}=\sum \frac{\left({O}_{i}-E_{i}\right)^{2}}{E_{i}} $$

\\(\chi^2\\)	=	chi squared

\\({O}_i\\)	=	observed value

\\(E_{i}\\)	=	expected value

### Spearman Correlation

In statistics, Spearman's rank correlation coefficient or Spearman's ρ, named after Charles Spearman and often denoted by the Greek letter or as, is a nonparametric measure of rank correlation. 

A Spearman Correlation test gives the strength of the association between two ordinal variables. It does not rely on normally distributed data. It assesses how well the relationship between two variables can be described using a monotonic function.

Formula:

$$ \rho=1-\frac{6 \sum d_{i}^{2}}{n (n^{2}-1)} $$

\\(\rho\\)	=	Spearman's rank correlation coefficient

\\(d_{i}\\)	=	difference between the two ranks of each observation

n	=	number of observations

### ANOVA Test

The difference between group means after any other variance in the outcome variable is accounted for

### Paired T-Test

A Paired T-test, sometimes called the dependent sample t-test, tells us the difference between two variables from the same population (ex: a before and after-test score for a group of people). In simpler words, it is used to determine whether the mean difference between two sets of observations is zero. In this test, each entity is accessed two times, resulting in pairs of observations and the difference is calculated. Then, a 1-sample t-test is performed on the results.

### Independent T-Test

The Independent T-Test, also called the two sample t-test, independent-samples t-test or student's t-test, is an inferential statistical test to find the difference between the same variable from different populations (ex: finding differences between males and females).

### Mann-Whitney U test

The Mann-Whitney U test is also called as the Mann–Whitney–Wilcoxon (MWW), Wilcoxon rank-sum test, or Wilcoxon–Mann–Whitney test. This is the non-parametric version of the popular t-test. It is used to compare differences between two independent groups when the dependent variable is either ordinal or continuous, but not normally distributed. For example, comparing the speed of people from two different groups in 100m race, where one group has trained for 4 weeks and the other has not. This test requires 

 - Two random, independent samples to compare.
 - The data is continuous - it must be possible to distinguish between values at the nth decimal place.
 - Scale of measurement should be ordinal, interval or ratio.
 
Null Hypothesis

The null hypothesis asserts that the medians of the two samples are identical.

Alternate Null Hypothesis

The alternative hypothesis says that one distribution is stochastically greater than the other. Or, All the observations from both groups are independent of each other.

### Simple Regression
How change in the predictor variable predicts the level of change in outcome variable.

### Multiple Regression
How changes in the combination of two or more predictor variables predict the level of change in outcome variables.
