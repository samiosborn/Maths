# plot_paths.py
import numpy as np
import matplotlib.pyplot as plt

# Plot several paths
def plot_paths(times, paths, title = None):
    # Set figure
    plt.figure(figsize=(6,4))
    # Plot paths
    n_paths, _ = np.shape(paths)
    for i in range(n_paths):
        plt.plot(times, paths[i], lw = 1.2)
    # Set title
    plt.title("Paths over time" if title is None else title)
    # Label axes
    plt.xlabel("t")
    plt.ylabel("Position")
    # Add grid
    plt.grid(True, alpha = 0.3)
    # Tight layout
    plt.tight_layout()   
    # Show plot
    plt.show()