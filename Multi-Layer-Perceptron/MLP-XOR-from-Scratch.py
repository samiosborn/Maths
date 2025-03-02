# Shallow MLP (feed-forward neural network) for XOR classification from scratch

import math, random
import numpy as np
import matplotlib.pyplot as plt

# XOR Classification Dataset
xor_test_inputs = [
    [0, 0],
    [0, 1], 
    [1, 0],
    [1, 1]
]

xor_test_outputs = [
    [0],
    [1],
    [1],
    [0]
]

# Initialising weights with Xavier initialisation (as sigmoid activation)
def initialise_weights(input_size, output_size):
    # Xavier initialisation formula
    weight_range = math.sqrt(6 / (input_size + output_size))
    weights = [random.uniform(-weight_range, weight_range) for i in range(input_size)]
    return weights

# Bias initialised to zero
def initialise_bias(size):
    return [0] * size

# Initialise weights and biases for the input layer sum
input_weights = initialise_weights(2, 1)    # 2 inputs (w1, w2), 1 output node
input_bias = initialise_bias(1)             # 1 bias term (b)

# Initialise weights and biases for the hidden layer
hidden_weights = initialise_weights(2, 1)   # 2 hidden nodes (v1, v2), 1 output node

# Input Layer Summation Function (s)
def input_layer_sum(x_inputs, input_weights, input_bias):
    return x_inputs[0]*input_weights[0] + x_inputs[1]*input_weights[1] + input_bias[0]

# Differentiation Input Layer Summation w.r.t. Input bias (b)
def d_input_layer_sum_d_input_bias(): 
    return 1

# Derivative of Input Layer Sum w.r.t. Weight 1 (w1)
def d_input_layer_sum_d_w1(x_inputs):
    return x_inputs[0]

# Derivative of Input Layer Sum w.r.t. Weight 2 (w2)
def d_input_layer_sum_d_w2(x_inputs):
    return x_inputs[1]

# Hidden Layer Quadratic Function (t)
def hidden_layer_transform(s, hidden_weights):
    return s * (s * hidden_weights[0] + hidden_weights[1])

# Derivative of Hidden Layer Quadratic w.r.t. s
def d_hidden_layer_transform_d_s(s, hidden_weights):
    return 2 * s * hidden_weights[0] + hidden_weights[1]

# Derivative of Hidden Layer Activation w.r.t. second order weight (v1)
def d_hidden_layer_transform_d_v1(s):
    return s**2

# Derivative of Hidden Layer Activation w.r.t. first order weight (v2)
def d_hidden_layer_transform_d_v2(s):
    return s

# Sigmoid Activation Function Output (o)
def sigmoid(t):
    # Prevents extreme values
    t = max(min(t, 500), -500)
    return 1 / (1 + math.exp(-t))

# Derivative of the Sigmoid Activation Function (s) w.r.t. t
def d_sigmoid_d_t(t):
    sigmoid_val = sigmoid(t)
    return sigmoid_val * (1 - sigmoid_val)

# Forward Pass Function
def forward_pass(x_inputs, input_weights, input_bias, hidden_weights):
    # Compute Input Layer Sum
    input_sum = input_layer_sum(x_inputs, input_weights, input_bias)

    # Compute Hidden Layer Activation
    hidden_output = hidden_layer_transform(input_sum, hidden_weights)

    # Compute Final Output with Sigmoid Activation
    output = sigmoid(hidden_output)
    
    return output

# Compute Predictions for Given Dataset
def compute_predictions(input_data, input_weights, input_bias, hidden_weights):
    predictions = []
    for sample in input_data:
        predictions.append(forward_pass(sample, input_weights, input_bias, hidden_weights))
    return predictions

# Loss Function (Cross-Entropy Loss)
def loss(prediction, actual):
    return -(actual * math.log2(prediction) + (1 - actual) * math.log2(1 - prediction))

# Derivative of Loss (l) w.r.t. Output (o)
def d_loss_d_o(prediction, actual, epsilon=1e-10):
    # Clip to avoid 0 or 1
    prediction = max(min(prediction, 1 - epsilon), epsilon)
    return (-actual / prediction) + (1 - actual) / (1 - prediction)

# Backward Pass Function (Gradient Descent)
def backward_pass(actuals, x_inputs, input_weights, input_bias, hidden_weights):
    training_loss = 0
    # Loop over traning data
    for i in range(len(actuals)):
        y = actuals[i]
        
        # Forward Pass
        s = input_layer_sum(x_inputs, input_weights, input_bias)
        t = hidden_layer_transform(s, hidden_weights)
        o = sigmoid(t)

        # Derivatives
        d_l_d_o = d_loss_d_o(o, y)
        d_o_d_t = d_sigmoid_d_t(t)
        d_t_d_s = d_hidden_layer_transform_d_s(s, hidden_weights)
        d_t_d_v1 = d_hidden_layer_transform_d_v1(s)
        d_t_d_v2 = d_hidden_layer_transform_d_v2(s)
        d_s_d_b = d_input_layer_sum_d_input_bias()
        d_s_d_w1 = d_input_layer_sum_d_w1(x_inputs)
        d_s_d_w2 = d_input_layer_sum_d_w2(x_inputs)

        # Update weights and biases
        input_bias[0] -= l_r * d_l_d_o * d_o_d_t * d_t_d_s * d_s_d_b
        input_weights[0] -= l_r * d_l_d_o * d_o_d_t * d_t_d_s * d_s_d_w1
        input_weights[1] -= l_r * d_l_d_o * d_o_d_t * d_t_d_s * d_s_d_w2
        hidden_weights[0] -= l_r * d_l_d_o * d_o_d_t * d_t_d_v1
        hidden_weights[1] -= l_r * d_l_d_o * d_o_d_t * d_t_d_v2

        training_loss += loss(o, y)

    # Cross Entropy Mean-Loss
    return training_loss / len(actuals)

# Compute Accuracy Function (L1 Norm)
def compute_accuracy(predictions, actuals):
    l_1 = 0
    for i in range(len(actuals)):
        l_1 += abs(actuals[i][0] - predictions[i])
    return 1 - l_1 / len(actuals)

# Track weight updates for plot
tracked_loss = []
weight_updates = {
    "w1": [],
    "w2": [],
    "b": [],
    "v1": [],
    "v2": [],
}

# Training loop
num_iterations = 100000

# Learning rate
l_r = 0.001

# Training Loop with Tracking
for epoch in range(num_iterations):
    total_loss = 0

    # Forward pass
    for i in range(len(xor_test_outputs)):
        loss_value = backward_pass(xor_test_outputs[i], xor_test_inputs[i], input_weights, input_bias, hidden_weights)
        total_loss += loss_value

    # Track weight updates
    weight_updates["w1"].append(input_weights[0])
    weight_updates["w2"].append(input_weights[1])
    weight_updates["b"].append(input_bias[0])
    weight_updates["v1"].append(hidden_weights[0])
    weight_updates["v2"].append(hidden_weights[1])

    # Compute and print mean loss
    avg_loss = total_loss / len(xor_test_outputs)
    if (epoch+1) % 10000 == 0:
        print(f"Iteration {epoch+1}, Loss: {avg_loss}")

# Print final weight values from tracking dictionary
print("Final Tracked Weights: ")
for key in weight_updates:
    print(f"{key}: {weight_updates[key][-1]}")

# Compute final predictions of test data after training
final_predictions = compute_predictions(xor_test_inputs, input_weights, input_bias, hidden_weights)
print(f"Test Data Inputs: {xor_test_inputs}")
print(f"Training Final Outputs: {final_predictions}")

# Calculate accuracy after training
accuracy = compute_accuracy(final_predictions, xor_test_outputs)
print(f"Training Accuracy: {accuracy * 100:.2f}%")

# Testing on perturbed inputs (5%)
perturbed_inputs = [
    [0.05, 0.05],
    [0.05, 0.95],
    [0.95, 0.05],
    [0.95, 0.95]
]
perturbed_predictions = compute_predictions(perturbed_inputs, input_weights, input_bias, hidden_weights)
print(f"Perturbed Inputs: {perturbed_inputs}")
print(f"Perturbed Predictions: {perturbed_predictions}")
perturbed_accuracy = compute_accuracy(perturbed_predictions, xor_test_outputs)
print(f"Training Accuracy (Perturbed Inputs): {perturbed_accuracy * 100:.2f}%")

# Function to compute the plot points for the 3D hyperplane of the trained FFN
def compute_output_for_grid(x_grid, y_grid, input_weights, input_bias, hidden_weights):
    # Create a grid to hold the output values
    z_grid = np.zeros_like(x_grid)
    for i in range(len(x_grid)):
        for j in range(len(x_grid[i])):
            # Compute the output for each point in the grid
            x_input = [x_grid[i][j], y_grid[i][j]]
            output = forward_pass(x_input, input_weights, input_bias, hidden_weights)
            z_grid[i][j] = output
    return z_grid

# Generate a grid of values for x and y
x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)
x_grid, y_grid = np.meshgrid(x_values, y_values)

# Compute the z outputs for each point in the grid
z_grid = compute_output_for_grid(x_grid, y_grid, input_weights, input_bias, hidden_weights)

# Plotting in 3D the hyperplane
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis')

# Labels and title
ax.set_xlabel('X Input')
ax.set_ylabel('Y Input')
ax.set_zlabel('XOR Output')
ax.set_title('3D Hyperplane of XOR Output')

# Plot the weight updates over training
plt.figure(figsize=(10, 5))
for key in weight_updates:
    plt.plot(weight_updates[key], label=key)
plt.xlabel('Iterations')
plt.ylabel('Weight Values')
plt.title('Weight Updates Over Training')
plt.legend()

# Show the plot
plt.show()