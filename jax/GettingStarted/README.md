# Playing around with JAX

First steps with the [JAX library from Google: Autograd and XLA, brought together for high-performance machine learning research.](https://github.com/google/jax)

## Differentiation

JAX supports forward mode and reverse mode differentiation.
From [this stackoverflow question](https://math.stackexchange.com/questions/2195377/reverse-mode-differentiation-vs-forward-mode-differentiation-where-are-the-be):

> In order to compute the Jacobian (determinant of the matrix of all the first-order partial derivatives of a 
> vector-valued function) of an operation, we should multiply the Jacobian the Jacobians of all sub-operations
> together. The difference between forward and reverse differentiation is the order in which we multiply these
> Jacobians.

According to the operation, the two differentiations might require a different number of operations in order to
obtain the Jacobian. Using a hybrid scheme of forward and reverse differentiation therefore results in an optimization
of the number of operations needed to compute the Jacobian. 

