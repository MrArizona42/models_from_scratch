### Problem

Problem: use linear combinations of X to predict y. The idea that linear combination of X is a good way to describe y is our first assumption. Mathematically it can be represented as the Hypothesys Function: 

$$
\text{Iterative form}\\
y^{(i)} \approx h(\theta) = \sum_{j=0}^{d} (\theta_jx_j^{(i)})
$$


$$
\text{Vector form}\\
\vec{y} \approx h(\theta) = X*\theta
$$

where $X$ - a matrix $n*d$ with  
$i \in \{1 ... n\}$ - number of examples (rows)  
$j \in \{1 ... d\}$ - number of features (columns)

### Loss function

The second step - is to find **the best $\theta$**. There are many ways to describe what 'the best' means. Here are some of them:
1. Probabilistic approach / Maximum Likelihood.

Assumptions:
* Assumes that the errors follow a normal distribution.
* Assumes that the errors are independent and identically distributed (i.i.d.).
* Then we need to choose **the most likely** $\theta$ using Maximum Likelihood approach

Solution:
Minimization of Squared Error

2. Linear algebra approach

Assumptions:
* If $y$ is approximately is a linear combination of $X$, then $\theta$ should be the closest vector to vector $y$ in column space $X$

Solution:
Minimization of Squared Error

3. Gauss-Markov theorem

Assumption (basically, similar to probabilistic approach):
* Assumes that the errors have zero mean.
* Assumes that the errors have constant variance (homoscedasticity).
* Assumes that the errors are uncorrelated.
* Assumes that the errors are normally distributed.

Then what whould be the BLUE (Best Linear Unbiased Estimators) $\theta$ - meaning the $\theta$ with smallest variance?

Solution:
Minimization of Squared Error

Even though minimization of MSE is far from solving the exact buiseness problems, often it works.

$$
\text{Iterative form}\\
J(\theta) = \frac{1}{n} * \sum_{i=1}^{n} (h^{(i)}(\theta) - y^{(i)})^2\\
J(\theta) = \frac{1}{n} * \sum_{i=1}^{n} (\sum_{j=0}^{d} (\theta_{j}x_{j}^{(i)}) - y^{(i)})^2 \\
\text{Vector form:} \\
J(\theta) = \frac{1}{n} * (X*\theta - y)^T*(X*\theta - y)
$$

### Find the minimum

To find the minimum of J there are two ways:
* Find the derivative of J with respect to Theta, set it equal to zero and solve the equation for Theta. That's possible for linear regression and the solution is:
$$
\text{Iterative form}\\
\frac{\partial}{\partial{\theta_j}} J(\theta) = \frac{1}{n} * \sum_{i=1}^{n} \left(\sum_{j=0}^{d} \left(\theta_{j}x_{j}^{(i)} \right) - y^{(i)} \right) * x_{j}^{(i)} \\
\text{Vector form:} \\
\nabla J(\theta) = \frac{1}{n} * (X \theta - y) * X \\
\nabla J(\theta) = 0 \\
\frac{1}{n} * \left(X \theta - y \right) * X = 0\\
\frac{1}{n} * X^TX\theta - \frac{1}{n} * X^Ty = 0 \\
\text{Finally:} \\
\theta = \left(\frac{1}{n} * X^T X \right)^{-1}*\frac{1}{n} * X^T y
$$
* Second way - find the derivative of J with respect to Theta and use gradient descent to find (almost) the minimum.

### Gradient Descent.

Update rule for Theta will be:

$$
\text{Iterative form}\\
\theta_j := \theta_j - \alpha * \frac{\partial}{\partial{\theta_j}} J(\theta) \\
\theta_j := \theta_j - \alpha * \frac{1}{n} * \sum_{i=1}^{n} \left(\sum_{j=0}^{d} \left(\theta_{j}x_{j}^{(i)} \right) - y^{(i)} \right) * x_{j}^{(i)} \\
\text{Vector form:} \\
\theta := \theta - \alpha * \nabla J(\theta) \\
\theta := \theta - \alpha * \frac{1}{n} * \left(X * \theta - y \right) * X
$$

### Regularization

**Important note: $\theta_0 \text{ / Intercept}$ should not be regularized!**

$$
\text{Iterative form}\\
J(\theta) = \frac{1}{n} * \sum_{i=1}^{n} (h^{(i)}(\theta) - y^{(i)})^2 + \lambda * ||\theta||_k^k\\
J(\theta) = \frac{1}{n} * \sum_{i=1}^{n} (\sum_{j=0}^{d} (\theta_{j}x_{j}^{(i)}) - y^{(i)})^2  + \lambda * ||\theta||_k^k\\
\text{Vector form:} \\
J(\theta) = \frac{1}{n} * (X*\theta - y)^T*(X*\theta - y) + \lambda * ||\theta||_k^k\\
$$

We just limit the absolute value of $\theta$ using its k-norm. Then for $k = 2$:

$$
\text{Iterative form}\\
\frac{\partial}{\partial{\theta_j}} J(\theta) = \frac{2}{n} * \sum_{i=1}^{n} (\sum_{j=0}^{d} (\theta_{j}x_{j}^{(i)}) - y^{(i)}) * x_{j}^{(i)}) + 2 * \lambda * \theta_j\\
\text{Vector form:} \\
\nabla J(\theta) = \frac{2}{n} * \left(X \theta - y \right) * X  + 2 * \lambda * \theta\\
$$

Exact solution for $k = 2$:

$$
\theta = \left(\frac{1}{n} * X^T X + \lambda * I \right)^{-1}*\frac{1}{n} * X^T y
$$

Gradient solution:

$$
\theta := \theta - \alpha * \nabla J(\theta) \\
\theta := \theta - \alpha * \left( \frac{2}{n} * \left(X \theta - y \right) * X  + 2 * \lambda * \theta \right)
$$

