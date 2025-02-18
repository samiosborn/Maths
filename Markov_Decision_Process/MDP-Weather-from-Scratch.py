import matplotlib.pyplot as plt

# Transition probability matrix
P = [
    [0.7, 0.2, 0.2],
    [0.2, 0.6, 0.3],
    [0.1, 0.2, 0.5]
]

# Initial state
initial_state = [1.0, 0.0, 0.0]
weather_states = ["Sunny", "Cloudy", "Rainy"]

# Function to compute the next state
def next_state(transition_matrix, current_state):
    return [
        sum(transition_matrix[i][j] * current_state[j] for j in range(len(current_state)))
        for i in range(len(transition_matrix))
    ]

# Function to compute state difference (L2 norm)
def state_difference(previous_state, current_state):
    return sum((previous_state[i] - current_state[i]) ** 2 for i in range(len(previous_state))) ** 0.5

# Convergence settings
epsilon = 0.01
max_iterations = 100

# Store states over time for plotting
states_over_time = [initial_state]

# Iteratively compute future states
current_state = initial_state
for iteration in range(max_iterations):
    next_state_vector = next_state(P, current_state)
    states_over_time.append(next_state_vector)
    # Checking how similar the two states are
    if state_difference(current_state, next_state_vector) < epsilon:
        break
    
    current_state = next_state_vector

# Transpose state history for plotting
state_trajectories = list(zip(*states_over_time))

# Plot the state evolution
plt.figure(figsize=(8, 5))
for i, state_values in enumerate(state_trajectories):
    plt.plot(state_values, label=weather_states[i])

plt.xlabel("Iterations")
plt.ylabel("Probability")
plt.title("Weather Steady-State Probabilities Convergence")
plt.legend()
plt.grid()
plt.show()

print(f"The steady state is: {current_state}, identified after {iteration} iterations.")
