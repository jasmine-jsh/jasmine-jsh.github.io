---
layout: post
title:  "How to Make a Linear Regression Model on PyTorch"
date:   2020-09-10 08:09:29 +0900
image:  04.png
tags:   MachineLearning
---

The simplest model of regression problem is <strong>linear regression</strong>. Currently, I am practicing how to develop on [`PyTorch`][PyTorch], so, it is the first baby step!
<br />
When using `PyTorch`, everything is so simple. We are only required to import a library which is already implemented and open to everyone and to select a module that we need.

---
### 1) Prepare Data
Split the dataset into <strong>train set</strong>, <strong>validation set</strong>, and <strong>test set</strong>.

<ul>
<li>Train set is literally for training a model.</li>
<li>Validation set is for testing a model in the learning process.</li>
<li>Test set is for evaluating a model after the learning process.</li>
</ul>

Let's suppose that there is data sized 2400. I splited it into (1600, 400, 400) for (train, validation, test) set.

```python
train_X, train_Y = X[:1600, :],Y[:1600]
val_X, val_Y = X[1600:2000, :],Y[1600:2000]
test_X, test_Y = X[2000:, :],Y[2000:]
```
<br />

### 2) Define 3 Key Components
Import `PyTorch` and the library [`torch.nn`][torch.nn].
```python
import torch
import torch.nn
```
<br />

I need [`nn.Linear`][nn.Linear] for our hypothesis.

<div align="center">
$$H(X)=XW+b$$ 
$$(W \in R^{n \times m}, b \in R^{n}, H \in R^{n})$$
m : number of input features, n : number of examples
</div>

```python
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_feature=m, out_feature=1, bias=True)
    def forward(self, x):
    return self.linear(x)   #Return the predicted value for x

model = LinearModel()
wieght = model.linear.weight
bias = model.linear.bias
```
<br />

The cost fuction is Mean Squared Error(MSE). We can use the module [`nn.MSELoss`][nn.MSELoss].

<div align="center">
$$cost(W,b) = \frac{1}{m} \sum_{i=1}^m (H(x^{(i)})-y^{(i)})^2$$
</div>

```python
reg_loss = nn.MSELoss()
loss = reg_loss(pred_y, true_y)   #Get the loss between true value and predicted value
loss.backward()     #Calulate partial derivatives of the loss for each parameter 
```
 
<br />

Various types of optimization function are given in [`torch.optim`][torch.optim]. We use [`optim.SGD`][optim.SGD] (Stochastic Gradient Decent).

<div align="center">
$$W:=W-\alpha \frac{\partial}{\partial W}cost(W)$$
</div>

[Learning rate][Learning rate] is a tuning parameter in an optimization algorithm that determines the step size at each iteration while moving toward a minimum of a loss function. 

```python
import torch.optim as optim
 
lr = 0.005
optimizer - optim.SGD(model.parameters(), lr=lr)
optimizer.zero_grad()   #Initialize optimizaer to 0
```

<br />

### 3) Train a Model
Here are some notions that we must NOT get confused.

<ul>
<li>Batch : the complete dataset </li>
<li>Mini-batch : a small part of batch </li>
    <ul>
    <li>Mini-batch size : the number of examples that the algorithm learns in a single pass </li>
    </ul>
<li>In a single pass, the algorithm goes through forward and backward, then update the parameters </li>
<li>Iteration : the number of passes the algorithm </li>
<li>Epoch : the number of times a learning algorithm learns the complete dataset(batch) </li>
</ul>

<div align="center">
Number of Iterations <strong>in 1 Epoch</strong> = (Size of Data) / (Size of Mini-batch)
</div>

```python
# ============ Train ============ #
model.train()

input_x = torch.Tensor(train_X)
true_y = torch.Tensor(train_Y)
pred_y = model(input_x)

loss = reg_loss(pred_y.squeeze(), true_y)
loss.backward()
optimizer.step()    #Update the parameters

# ========== Validation ========== #
model.eval()

input_x = torch.Tensor(val_X)
true_y = torch.Tensor(val_Y)
pred_y = model(input_x)

loss = reg_loss(train_pred_y.squeeze(), train_true_y)
```

<br />

### 4) Evaluate the Model
When evaluating a model, we have to choose a metric. I used mean absolute error([`MAE`][MAE]) in the library [sklearn.metrics][sklearn.metrics].

<div align="center">
$$MAE(Y_{true}, Y_{predict}) = \sum_{i} | \ y_{true}^{(i)} - y_{predict}^{(i)} \ | $$
</div>

```python
from sklearn.metrics import mean_absolute_error

model.eval()

input_x = torch.Tensor(test_X)
true_y = torch.Tensor(test_Y)
pred_y = model(input_x)

mae = mean_absolute_error(true_y, pred_y)
```

<br />

### 5) Result
The figure(<em>left</em>) shows train loss and validation loss while learning a model. If validation loss keeps increasing, it means that there is no use in learning further.

The figure(<em>right</em>) shows MAE as the epoch goes by. MAE is continuously decreased until all epochs are done. I would increase the number of epochs to see what happens.


![]({{site.baseurl}}/images/05.png)


#### [Reference][Reference] 
<iframe src="https://youtu.be/-hWgqTB09DM" frameborder="0" allowfullscreen></iframe>


[PyTorch]: https://pytorch.org/
[torch.nn]: https://pytorch.org/docs/stable/nn.html
[nn.Linear]: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html?highlight=nn%20linear#torch.nn.Linear
[nn.MSELoss]: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
[torch.optim]: https://pytorch.org/docs/master/optim.html#torch-optim
[optim.SGD]: https://pytorch.org/docs/master/optim.html?highlight=sgd#torch.optim.SGD
[Learning rate]: https://en.wikipedia.org/wiki/Learning_rate
[MAE]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
[sklearn.metrics]: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
[Reference]: https://github.com/heartcored98/Standalone-DeepLearning