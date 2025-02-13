import matplotlib.pyplot as plt

# Newton-Raphson from scratch on a multivariate function
# f(x,y) = x^2 + y^2 + x*y + x + 2*y
def f(x,y):
    return x**2 + y**2 + x*y + x + 2*y

# Partial derivatives for gradient vector
def dfdx(x,y):
    return 2*x + y + 1

def dfdy(x,y):
    return 2*y + x + 2

# Starting points
start_x, end_x = -10.0, 10.0
start_y, end_y = -10.0, 10.0

# Track x,y for convergence - start at corner
x_est = [start_x]
y_est = [end_y]
f_est = [f(x_est[0],y_est[0])]

# Set learning rate
learning_rate = 0.1

# Set number of iterations
num_iterations = 100

# Loop update
for i in range(num_iterations):
    x_est.append(x_est[i] - learning_rate*dfdx(x_est[i],y_est[i]))
    y_est.append(y_est[i] - learning_rate*dfdy(x_est[i],y_est[i]))
    f_est.append(f(x_est[i],y_est[i]))

print("The minima of f is: ",f_est[num_iterations]," at point: ",[x_est[num_iterations],y_est[num_iterations]])

# Plotting f for graph
graph_resolution = 100

x_vals = [(i * (end_x - start_x) / (graph_resolution) + start_x) for i in range(graph_resolution)]
y_vals = [(i * (end_y - start_y) / (graph_resolution) + start_y) for i in range(graph_resolution)]

# Creating a list of points
x_plot = []
y_plot = []
f_plot = []

for x in x_vals:
    for y in y_vals:
        x_plot.append(x)
        y_plot.append(y)
        f_plot.append(f(x,y))


# Plot the 3D line
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Surface plot with a triangular mesh
ax.plot_trisurf(x_plot, y_plot, f_plot, cmap='viridis', alpha=0.7)

# Plot the Newton-Raphson descent path as a red line
ax.plot3D(x_est, y_est, f_est, color='red', linewidth=2, label="Newton-Raphson Path")

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Surface with Newton-Raphson descent path in red')

# Show legend
ax.legend()

# Show plot
plt.show()
