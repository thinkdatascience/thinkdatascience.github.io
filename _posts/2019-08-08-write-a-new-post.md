---
title: Writing a New Post
author: Cotes Chung
date: 2019-08-08 14:10:00 +0800
categories: [Blogging, Tutorial]
tags: [writing]
math: true
mermaid: true
---


## Detecting Multicollinearity using Variance Inflation Factor

Now that we know what multicollinearity is, how can we tell if it is present in our data ?
Variance Inflation Factors (VIF) is one of the most common techniques used to detect multicollinearity.
VIF measures the severity of multicollinearity in regression analysis

> VIF provides an index that measures how much the variance of an estimated regression coefficient is increased because of collinearity.

VIF is calculated as follows:

$$ x = {-b \pm \sqrt{b^2-4ac} \over 2a} $$

$$ VIF_j  =  {1 \over 1 - R_j^2} $$


Each variable \\(x_j\\) in the dataset is separately treated as the target variable and the remaining variables are treated as the predictors. Next, a linear model is fit 
and \\(R^2\\) value is calculated. Finally, VIF for the target variable is obtained using the above equation.

VIF is always positive and is high when \\(R^2\\) is closer to 1.  

### Interpreting \\(R^2//) and VIF


\\(R_j^2\\) value determines how well an independent variable is described by the other independent variables. 
When \\(R_j^2\\) value is equal to 0, the variance of the remaining independent variables cannot be predicted from the \\(j^th\\) independent variable. Therefore, when \\(R_j^2\\) = 0 (i.e VIF = 1) which implies that the \\(j^th\\) variable is not correlated to the remaining variabeles or in other words, multicollinearity does not exist in this regression model. In such case, the variance of \\(x_j\\) is not inflated at all. 


> *Note*: As a rule of thumb, a VIF greater than 4 indicates that multicollinearity might exist and further investigation is required. VIF greater than 10 implies a significant multicollinearity that needs to be corrected.













## Naming and Path

Create a new file named `YYYY-MM-DD-TITLE.EXTENSION` and put it in the `_post/` of the root directory. Please note that the `EXTENSION` must be one of `md` and `markdown`.

## Front Matter

Basically, you need to fill the [Front Matter](https://jekyllrb.com/docs/front-matter/) as below at the top of the post:

```yaml
---
title: TITLE
date: YYYY-MM-DD HH:MM:SS +/-TTTT
categories: [TOP_CATEGORIE, SUB_CATEGORIE]
tags: [TAG]     # TAG names should always be lowercase
---
```

> **Note**: The posts' ***layout*** has been set to `post` by default, so there is no need to add the variable ***layout*** in Front Matter block.

### Timezone of date

In order to accurately record the release date of a post, you should not only setup the `timezone` of `_config.yml` but also provide the the post's timezone in field `date` of its Front Matter block. Format: `+/-TTTT`, e.g. `+0800`.

### Categories and Tags

The `categories` of each post is designed to contain up to two elements, and the number of elements in `tags` can be zero to infinity. For instance:

```yaml
categories: [Animal, Insect]
tags: [bee]
```

## Table of Contents

By default, the **T**able **o**f **C**ontents (TOC) is displayed on the right panel of the post. If you want to turn it off globally, go to `_config.yml` and set the value of variable `toc` to `false`. If you want to turn off TOC for specific post, add the following to post's [Front Matter](https://jekyllrb.com/docs/front-matter/):

```yaml
---
toc: false
---
```

## Comments

Similar to TOC, the [Disqus](https://disqus.com/) comments is loaded by default in each post, and the global switch is defined by variable `comments` in file `_config.yml` . If you want to close the comment for specific post, add the following to the **Front Matter** of the post:

```yaml
---
comments: false
---
```

## Mathematics

For website performance reasons, the mathematical feature won't be loaded by default. But it can be enabled by:

```yaml
---
math: true
---
```

## Mermaid

[**Mermaid**](https://github.com/mermaid-js/mermaid) is a great diagrams generation tool. To enable it on your post, add the following to the YAML block:

```yml
---
mermaid: true
---
```

Then you can use it like other markdown language: surround the graph code with <code class="highlighter-rouge">```mermaid</code>.

## Images

### Preview image

If you want to add an image to the top of the post contents, specify the url for the image by:

```yaml
---
image: /path/to/image-file
---
```

### Image caption

Add italics to the next line of an image，then it will become the caption and appear at the bottom of the image:

```markdown
![img-description](/path/to/image)
_Image Caption_
```

### Image size

You can specify the width (and height) of a image with `width`:

```markdown
![Desktop View](/assets/img/sample/mockup.png){: width="400"}
```

### Image position

By default, the image is centered, but you can specify the position by using one of class `normal` , `left` and `right`. For example:

- **Normal position**

  Image will be left aligned in below sample:

  ```markdown
  ![Desktop View](/assets/img/sample/mockup.png){: width="350" class="normal"}
  ```

- **Float to the left**

  ```markdown
  ![Desktop View](/assets/img/sample/mockup.png){: width="240" class="left"}
  ```

- **Float to the right**

  ```markdown
  ![Desktop View](/assets/img/sample/mockup.png){: width="240" class="right"}
  ```

> **Limitation**: Once you specify the position of an image, it is forbidden to add the image caption.

## Pinned Posts

You can pin one or more posts to the top of the home page, and the fixed posts are sorted in reverse order according to their release date. Enable by:

```yaml
---
pin: true
---
```

## Code Block

Markdown symbols <code class="highlighter-rouge">```</code> can easily create a code block as following examples.

```
This is a common code snippet, without syntax highlight and line number.
```

## Specific Language

Using <code class="highlighter-rouge">```language</code> you will get code snippets with line numbers and syntax highlight.

> **Note**: The Jekyll style `{% raw %}{%{% endraw %} highlight LANGUAGE {% raw %}%}{% endraw %}` or `{% raw %}{%{% endraw %} highlight LANGUAGE linenos {% raw %}%}{% endraw %}` are not allowed to be used in this theme !

```yaml
# Yaml code snippet
items:
    - part_no:   A4786
      descrip:   Water Bucket (Filled)
      price:     1.47
      quantity:  4
```

### Liquid Codes

If you want to display the **Liquid** snippet, surround the liquid code with `{% raw %}{%{% endraw %} raw {%raw%}%}{%endraw%}` and `{% raw %}{%{% endraw %} endraw {%raw%}%}{%endraw%}` .

{% raw %}
```liquid
{% if product.title contains 'Pack' %}
  This product's title contains the word Pack.
{% endif %}
```
{% endraw %}

## Learn More

For more knowledge about Jekyll posts, visit the [Jekyll Docs: Posts](https://jekyllrb.com/docs/posts/).

