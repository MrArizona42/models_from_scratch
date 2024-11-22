### Linear combinations >> probability

The idea is similar to Linear Regression - use $X*\theta$ for prediction. But now we say that we want to predict the probability - the value $P(y | X) \in [0 ... 1]$

$$
P(y = 1|x;\theta) = h_\theta(x)
$$
$$
P(y = 0|x;\theta) = 1 - h_\theta(x)
$$

We need some $g(\theta^T * X)$
$$
h_\theta(X) = g(\theta^T * X) = g(z), \text{ where } z = \theta^T * X \in [-\infty ... +\infty]
$$
In other words, we need:
$$
\theta^T * X \in [-\infty ... +\infty] =>> P(y | X) \in [0 ... 1]
$$

- transform $\theta^T * X$ into $[0 ... 1]$
- or find some $f(P)$ that returns $[-\infty ... +\infty]$ and derive $P(y | X)$ from it.

### Odds and log(odds)

$$
P(y = 1|x;\theta) = h_\theta(x) \in [0 ... 1]
$$
$$
Odds(y = 1) = \frac {P(y = 1)} {P(y = 0)} = \frac {P(y = 1)} {1 - P(y = 1)}  \in [0 ... +\infty]
$$
$$
\log{Odds(y = 1)} = \log{\frac {P(y = 1)} {1 - P(y = 1)}} \in [-\infty ... +\infty]
$$
We found a function $f(P)$ that returns values $[-\infty ... +\infty]$. So, now we can make aт assumption that
$$
\theta^T * X = \log{\frac {P(y = 1)} {1 - P(y = 1)}}
$$
Then:
$$
P(y = 1 | X; \theta) = h_\theta(X) = \frac {1} {1 + e^{-(\theta^T * X)}}
$$

Now the question is - how to find the best $\theta$

### Likelihood



$$
\text{For } y = 1: p(y|x,\theta) = h_\theta(x)
$$
$$
\text{For } y = 0: p(y|x,\theta) = 1 - h_\theta(x)
$$
Same thing with one formula:
$$
p(y|x,\theta) = (h_\theta(x))^y (1 - h_\theta(x))^{1 - y}
$$
$$
p(y|x,\theta) = (\frac {1} {1 + e^{-(\theta^T * X)}})^y (1 - \frac {1} {1 + e^{-(\theta^T * X)}})^{1 - y}
$$

We need the most likely parameters theta. The LIKELIHOOD of parametrs $\theta$ is basically the same as the probability of y but with $\theta$ as a variable. Probability of all y values is the product of all probabilities (joint probability).

$$
L(\theta) = p(\vec{y}|X;\theta)
$$
$$
= \prod_{i=1}^{n} p(y^{(i)}|x^{(i)};\theta)
$$
$$
= \prod_{i=1}^{n} (h_\theta(x^{(i)}))^{y^{(i)}} (1 - h_\theta(x^{(i)}))^{1 - y^{(i)}}
$$
$$
= \prod_{i=1}^{n} (\frac {1} {1 + e^{-(\theta * x^{(i)})}})^y (1 - \frac {1} {1 + e^{-(\theta * x^{(i)})}})^{1 - y}
$$

Logistic function is monotonic, which allows us to maximize the logarythm of this product. We replace Linelihood with LOG of Likelihood. Then PRODUCT will transform to SUMM.

$$
l(\theta) = \log L(\theta)
$$
$$
= \sum_{i=1}^n y^{(i)} \log h(x^{(i)}) + (1 - y^{(i)})log(1 - h(x^{(i)})
$$
$$
= \sum_{i=1}^n y^{(i)} \log \frac {1} {1 + e^{-(\theta * x^{(i)})}} + (1 - y^{(i)})log(1 - \frac {1} {1 + e^{-(\theta * x^{(i)})}})
$$

But now the value depends on the dataset size - sum from 1 to n. So, for convenience we need to scale the whole equation:

$$
l(\theta) = \frac{1}{n} * \sum_{i=1}^n y^{(i)} \log \frac {1} {1 + e^{-(\theta * x^{(i)})}} + (1 - y^{(i)})log(1 - \frac {1} {1 + e^{-(\theta * x^{(i)})}})
$$

And this thing we want to maximize

### Derivative of Likelihood function

In advance we can say:

$$
g(z) = \frac{1}{1 + e^{-z}}
$$
Derivative of a fraction of two functions
$$
\frac{d}{dz} g(z) = \frac{d}{dz} \frac{1}{f(z)} =
$$
$$
= \frac{0 * (1 + e^{-z}) - 1 * \frac{d}{dz} (1 + e^{-z}) )}{(1 + e^{-z})^2} =
$$
$$
= \frac{-(-e^{-z})}{(1 + e^{-z})^2} = \frac{e^{-z} + 1 - 1}{(1 + e^{-z})^2} =
$$
$$
= (1 - \frac{1}{1 + e^{-z}}) * \frac{1}{1 + e^{-z}} =
$$
$$
= g(z) * (1 - g(z))
$$
Coming back to $h_{\theta}(x)$ we need to add $\frac{dz}{dx} = \frac{d\theta^T*X}{dx} = X$ to the end
$$
\text{1. } \frac {\partial{}} {\partial{\theta}} h_\theta(X) = h_\theta(X) * (1 - h_\theta(X)) * X = \frac {1} {1 + e^{-(\theta^T * X)}} * (1 - \frac {1} {1 + e^{-(\theta^T * X)}}) * X
$$
$$
\text{2. } \frac {\partial{}} {\partial{\theta}} \log{h_\theta(X)} = \frac{1}{h_\theta(X)} * h_\theta(X) * (1 - h_\theta(X) * X) = (1 - h_\theta(X)) * X
$$
$$
\text{3. } \frac {\partial{}} {\partial{\theta}} \log{\left(1 - h_\theta(X) \right)} = \frac{1}{1 - h_\theta(X)} * \left[ 0 - h_\theta(X) * (1 - h_\theta(X)) * X \right]  =
$$
$$
= \frac{1}{1 - h_\theta(X)} * 0 - \frac{1}{1 - h_\theta(X)} * h_\theta(X) * (1 - h_\theta(X)) * X = \\
= - h_\theta(X) * X
$$

Finally:

$$
\frac {\partial{}} {\partial{\theta}} l(\theta) = \frac{1}{n} * \sum_{i = 1}^{n} X * \left( y^{(i)} - h_\theta(X) \right) = \frac{1}{n} * \sum_{i = 1}^{n} X * \left( y^{(i)} - \frac {1} {1 + e^{-(\theta * x^{(i)})}} \right)
$$

Gradient solution:

$$
\theta := \theta + \alpha * \frac {\partial{}} {\partial{\theta}} l(\theta)
$$

### Regularization

Works the same as for Linear Regression:

$$
\text{old l}(\theta) = \frac{1}{n} * \sum_{i=1}^n y^{(i)} \log \frac {1} {1 + e^{-(\theta * x^{(i)})}} + (1 - y^{(i)})log(1 - \frac {1} {1 + e^{-(\theta * x^{(i)})}})
$$
$$
\text{new l}(\theta) = \text{old l}(\theta) + \lambda * ||\theta||_k^k
$$

For k=2:
$$
\text{new l}(\theta) = \text{old l}(\theta) + \lambda * ||\theta||_2^2
$$
$$
\frac {\partial{}} {\partial{\theta}} \text{new l}(\theta) = \frac {\partial{}} {\partial{\theta}} \text{old l}(\theta) + 2 * \lambda * \theta =
$$
$$
= \frac{1}{n} * \sum_{i = 1}^{n} X * \left( y^{(i)} - \frac {1} {1 + e^{-(\theta * x^{(i)})}} \right) + 2 * \lambda * \theta
$$

$$
\theta := \theta + \alpha * \frac {\partial{}} {\partial{\theta}} \text{new l}(\theta) =
$$
$$
\theta := \theta + \alpha * \left( \frac{1}{n} * \sum_{i = 1}^{n} X * \left( y^{(i)} - \frac {1} {1 + e^{-(\theta * x^{(i)})}} \right) + 2 * \lambda * \theta \right)
$$