---
title: Bias-Variance Tradeoff
author: Akshay Adlakha & Akshaykumar Rao
date: 2020-12-14 00:34:00 +0800
categories: [Blogging, Tutorial]
tags: [favicon]
toc: false
---

The errors associated with machine learning models are of two types:
- Bias
- Variance

It is important to understand these two errors and the tradeoff to minimize the total error and avoid Underfitting & Overfitting of the model.

## Bias

Bias means that the model favors one outcome more than the others and how far the model prediction from the correct prediction. It occurs due to simpler assumptions made by the model to make the predictions. The model with high bias pays very less attention to training data and oversimplies the model. In other words, the model leads to underfitting i.e, a high training and a high testing error. 
## Variance

Variance reflects the spreadness. How much the predictions for a given point vary between different realizations of the model. The model with high variance pays much attention to traning data i.e, tries to capture everything from the training data. When it comes to testing data, it doesn't generalize well. So, such model leads to overfitting i.e. a low training error but high testing error.

So, it is neccessary to balance between bias and variance. We don't want to have a model with high errors. A balance between bias and variance minimizes the total error of the model. This balance is known as Bias-Variance tradeoff.






In [**Chirpy**](https://github.com/cotes2020/jekyll-theme-chirpy/), the image files of [Favicons](https://www.favicon-generator.org/about/) are placed in `assets/img/favicons/`. You may need to replace them with your own. So let's see how to customize these Favicons.

With a square image (PNG, JPG or GIF) in hand, open the site [*Favicon & App Icon Generator*](https://www.favicon-generator.org/) and upload your original image.

![upload-image](/assets/img/sample/upload-image.png)

Click button <kbd>Create Favicon</kbd> and wait a moment for the website to generate the icons of various sizes automatically.

![download-icons](/assets/img/sample/download-icons.png){: width="600"}

Download the generated package, unzip and delete the following two from the extracted files:

- browserconfig.xml
- manifest.json
 
Now, copy the remaining image files (`.PNG` and `.ICO`) from the extracted `.zip` file to cover the original files in the folder `assets/img/favicons/`.

The following table helps you understand the changes to the icon file:

> ✓ means keep, ✗ means delete.

| File(s)             | From Favicon & App Icon Generator | From Chirpy |
|---------------------|:---------------------------------:|:-----------:|
| `*.PNG`             | ✓                                 | ✗           |
| `*.ICO`             | ✓                                 | ✗           |
| `browserconfig.xml` | ✗                                 | ✓           |
| `manifest.json`     | ✗                                 | ✓           |


The next time you build the site, the icon will be replaced with a customized edition.
