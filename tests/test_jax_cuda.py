import jax
import jax.numpy as jnp

# Check available devices
print("Available devices:", jax.devices())

# Run a simple computation on GPU
def simple_computation():
    x = jnp.arange(10)
    return jnp.sum(x)

# Use jit to compile and run on GPU
jitted_computation = jax.jit(simple_computation)
result = jitted_computation()

print("Computation result:", result)
print("Computation device:", jax.devices()[0])
