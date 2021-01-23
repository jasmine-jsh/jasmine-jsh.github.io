---
layout: post
title:  "3 Key Components of ML"
date:   2020-09-07 14:30:41 +0900
image:  03.png
tags:   MachineLearning
---

## Hypothesis
A candidate <strong>model</strong> that approximates a target function for mapping inputs to outputs and make predictions on unknown data

## Cost Function
The function we want to minimize or maximize \
(We may also call it the <strong>loss function</strong> or <strong>error function</strong>.)

## Optimization
The task of either minimizing or maximizing the cost function $$f(x)$$ by altering $$x$$

<br />
<br />
In machine learning, specifically supervised learning, we can define these 3 key components as follows :


---
### 1) Linear Regression

<ul>
<li> <strong>Hypothesis</strong> $$H(x)=Wx+b$$ </li>

<li> <strong>Cost</strong> $$cost(W,b) = \frac{1}{m} \sum_{i=1}^m (H(x^{(i)})-y^{(i)})^2$$ </li>

<li> <strong>Optimization</strong> $$W:=W-\alpha \frac{\partial}{\partial W}cost(W)$$ </li>
</ul>


---
### 2) Classification

<ul>
<li> <strong>Hypothesis</strong> $$H(x)=Wx+b$$ </li>

<li> <strong>Cost</strong> $$cost(W,b) = \frac{1}{m} \sum_{i=1}^m (H(x^{(i)})-y^{(i)})^2$$ </li>

<li> <strong>Optimization</strong> $$W:=W-\alpha \frac{\partial}{\partial W}cost(W)$$ </li>
</ul>
