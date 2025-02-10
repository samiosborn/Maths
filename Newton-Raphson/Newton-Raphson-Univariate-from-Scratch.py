# Newton–Raphson gradient descent on a univariate function
# Goal: Identify x_min which minimises f

import math
import matplotlib.pyplot as plt

# Define a univariate function f
# f = x^2 - sin(x)
def f(x):
    return x**2 - math.sin(x)    

# Differentiate f w.r.t. x to df_dx
# df/dx = 2*x - cos(x)
def grad_f(x):
    return 2*x - math.cos(x)

# Define a range over x-axis where the minima is contained [0,1]
num_points = 1000
X_vals_start = -1.000
X_vals_end = 1.000
X_vals = [X_vals_start]

for j in range(num_points):
    X_vals.append(((X_vals_end - X_vals_start) * (1/num_points))+X_vals[j])

Y_vals = [f(x) for x in X_vals]
print(len(X_vals),len(Y_vals))

# Starting x as 0
x_est = [0]

# Starting f using x_0
f_est = [f(x_est[0])]

# Define learning rate
learning_rate = 0.1

# Defining the number of iterations
num_iterations = 100

# Updating x using NR algorithm
for i in range(num_iterations):
    x_est.append(x_est[i] - learning_rate * grad_f(x_est[i]))
    f_est.append(f(x_est[i+1]))

print(x_est[num_iterations],f_est[num_iterations])

# Plot two lines on the same graph
fig = plt.figure()

# Adding axes as part of subplot arrangement
ax = fig.add_subplot(1,1,1)

# Plotting f and the convergence to minima
ax.plot(X_vals, Y_vals, color='tab:blue',linewidth=1)
ax.plot(x_est,f_est,color='tab:orange',linewidth=2, linestyle = 'dashed')
ax.plot(x_est[num_iterations],f_est[num_iterations],color='tab:red',marker='o')
plt.show()