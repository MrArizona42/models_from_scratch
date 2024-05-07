# Decision trees

Building a decision tree using the simplest iterative approach.
* Information criterion. What does it mean to make a good split?
$$
H(X_m) - \frac{|X_l|}{|X_m|}H(X_l) - \frac{|X_r|}{|X_m|}H(X_r) => max
$$
* Explanation of Entropy
$$
H(X) = - \sum^{k=1}_{K}p_k \log p_k
$$
* and Gini impurity.
$$
H(X) = \sum^{k=1}_{K} p_k (1 - p_k) = 1 - \sum^{k=1}_{K} p^2_k
$$
* Target encoding
* Stopping criteria.
* Overfitting visualization.