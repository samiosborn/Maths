# plot_brownian_paths.py
from brownian_paths import brownian_paths
from plot_paths import plot_paths

# Set parameters
T = 1.0
n_steps = 1000
n_paths = 30

# Simulate Brownian paths
t, W = brownian_paths(n_paths=n_paths, n_steps=n_steps, T=T, seed=123)

# Plot them
plot_paths(t, W, title="Standard Brownian Motion")
