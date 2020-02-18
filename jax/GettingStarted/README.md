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

## Using XLA to compile code

JAX leverages [XLA](https://www.tensorflow.org/xla) in order to compile the code to run on accelerators rather than on
CPU. XLA uses JIT compilation by default for NumPy operations, and lets the user use JIT to compile user-made Python
functions into XLA-optimized kernels.

## Grad and VMap

JAX isn't simply a NumPy on GPU. In addition to the aforementioned JIT compilation, it also comes with two additional
programs:

- grad, which computes derivatives of arbitrary functions (accepts many Python statements such as if-else, loops, ...)
- vmap, which is a vectorizing map. Works similarly to a mapping function along an array's axis, but instead of
  keeping the loop on the outside of the operation (e.g. multiplying a matrix with an array of vectors), it pushes
  the loop down into a function's primitive operations for better performance (prioritizing a matrix by matrix
  operation rather than a loop of matrix by vector operations, for example).