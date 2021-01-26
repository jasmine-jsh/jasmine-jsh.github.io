---
layout: post
title:  "What is Perceptron?"
date:   2020-09-16 16:22:7 +0900
image:  06.png
tags:   DeepLearning
---

A <strong>perceptron</strong> is a mathmatical model of a biological neuron. It is a unit of a neural network and also is regarded as <strong>a single layer neural network</strong>. A multy-layer perceptron is called Neural Networks. If you want to know how neural network works, it is necessary to understand how a perceptron works.

---
### A Neuron
Human brain is a collection of billions of neurons.

![]({{site.baseurl}}/images/07.png)

<ol>
<li> <strong>Dendrites</strong> receive information from other neurons. </li>
<li> <strong>Cell nucleus(Soma)</strong> processes the information. </li>
<li> <strong>Axon</strong> sends information. </li>
</ol>
<ul>
<li> <strong>Synapse</strong> is the connection between an axon terminal and other neuron dendrites. </li>
</ul>


An output signal is transmitted only when the total strength of the input signals is larger than a certain threshold. People modeled this phenomenon in a perceptron by calculating the weighted sum of the input signals and applying an activation function on the sum to determine its output.

---
### A Perceptron
Neural network is a collection of perceptrons.

![]({{site.baseurl}}/images/08.png)

<ol>
<li> <strong>Inputs</strong> are given in the form of a vector. </li>
<li> The <strong>weighted sum</strong> is calculated by multiplying the inputs and their weights. </li>
<li> <strong>Output</strong> is produced by applying a nonlinear function to the sum. </li>
</ol>

It is an algorithm for supervised learning of binary classifier. The perceptron learns the weights for input signals <strong>in order to draw a linear decision boundary</strong>.

<div align="center">
$$\begin{equation}
            f(x)=
            \begin{cases}
                1 & \text{if } \mathbf{w}^\intercal \mathbf{x} + \mathbf{b} > 0 \\
                0 & \text{otherwise}\\
            \end{cases}       
        \end{equation}$$
</div>
![]({{site.baseurl}}/images/10.png)


The <strong>activation function</strong> applies a step rule to check if the output is greater than zero or not. Mostly, non-linear fuctions (differentiable and monotonic) are used.

![]({{site.baseurl}}/images/09.png)
