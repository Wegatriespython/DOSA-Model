import jax
import jax.numpy as jnp

def f(x):
    return x**2 + jnp.sin(x) * jnp.log(x + 1)

# Calculate the derivative
grad_f = jax.grad(f)
x_val = 2.0
gradient_at_x = grad_f(x_val)  # Automatically calculated derivative at x_val
