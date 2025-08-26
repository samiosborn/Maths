# brownian_paths.py
import numpy as np

# Simulate paths of Brownian motion
def brownian_paths(n_paths, n_steps, T, seed = None):
    # If Seed not passed
    if seed is not None:
        # Set the global NumPy RNG seed
        np.random.seed(seed)

    # Time delta
    dt = T / n_steps

    # Time array
    t = np.linspace(0.0, T, n_steps + 1)

    # Sample stanard normal incremenets for all paths and steps 
    Z = np.random.randn(n_paths, n_steps)
    # Scale increments by sqrt(dt)
    dW = np.sqrt(dt) * Z

    # Initialise Wiener process paths
    W = np.zeros((n_paths, n_steps + 1))

    # Paths as cumulative sums of increments 
    W[:, 1:] = np.cumsum(dW, axis = 1)

    return t, W