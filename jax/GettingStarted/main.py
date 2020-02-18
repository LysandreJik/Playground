from jax import grad
import jax.numpy as np


def tanh(x):
	y = np.exp(-2.0 * x)
	return (1.0 - y) / (1.0 + y)


def main():
	grad_tanh = grad(tanh)
	print(grad_tanh(1.0))


if __name__ == "__main__":
	print("JAX playground")
	main()
