# Deterministic MDP maze-grid (finite-state) solver with value iteration and greedy optimal policy
import numpy as np
import matplotlib.pyplot as plt

# Define an 8x8 maze where 1 represents a wall and 0 represents a path
maze = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 1],
    [0, 1, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1]
])

# Define the start and goal positions
start_position = (3, 0)  # Starting position in the maze (not needed here)
goal_position = (4, 7)   # Goal position in the maze

# Define possible actions (Up, Down, Left, Right) as a list of tuples (immutable)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Dictionary for mapping actions to ASCII symbols
action_symbols = {(-1, 0): "↑", (1, 0): "↓", (0, -1): "←", (0, 1): "→"}

# Discount factor for future rewards
gamma = 0.9  # Determines the importance of future rewards

# Initialise the rewards matrix (constant)
rewards = np.full(maze.shape, -1)   # Penalise each step to encourage shorter paths
rewards[goal_position] = 1000       # Large reward for reaching the goal
rewards[maze == 1] = -100           # Large penalty for hitting a wall

# Initialise the value function (all zeros initially)
value_function = np.zeros(maze.shape)

# Initialise policy grid for printing to console
policy_grid = np.full(maze.shape, " ", dtype=str)
# No policy for positions (states) that are either walls or the goal
for row in range(maze.shape[0]):
    for column in range(maze.shape[1]):
        if maze[row, column] == 1:
            policy_grid[row, column] = "■"  # Wall
        elif (row, column) == goal_position:
            policy_grid[row, column] = "G"  # Goal

# Initialise greedy policy matrix
greedy_policy = np.full(maze.shape, None, dtype=object)

# Store the value matrix at each iteration for visualisation later
value_log = []

# Check if a position is within the maze path and not a wall (or outside)
def is_valid_position(row, column):
    return (0 <= row < maze.shape[0]) and (0 <= column < maze.shape[1]) and (maze[row, column] != 1)

# Convergence threshold for value iteration
theta = 0.0001

# Value iteration until convergence
def value_iteration():
    global value_function, greedy_policy
    iteration = 0
    while True:
        delta = 0
        # Creating a copy of the value function for comparison purposes later
        new_value_function = np.copy(value_function)
        # Looping through all the states (positions) in the maze
        for row in range(maze.shape[0]):
            for column in range(maze.shape[1]):
                # Don't bother with values for positions which are walls or the goal
                if (row, column) == goal_position or maze[row, column] == 1:
                    continue
                # Initialise the possible actions from that position
                action_values = []
                # Initialise the best action value for each position
                best_value = float("-inf")
                # Looping through the set of actions
                for action in actions:
                    # The new position after said action
                    new_row, new_column = row + action[0], column + action[1]
                    # If the new position is invalid, stay in the current position
                    if not is_valid_position(new_row, new_column):
                        new_row, new_column = row, column  
                    # Compute the value of this action using the Bellman equation
                    action_value = rewards[new_row, new_column] + gamma * value_function[new_row, new_column]
                    # Storing the value for this action
                    action_values.append(action_value)
                    # Keep track of the best action
                    if action_value > best_value:
                        best_value = action_value
                        greedy_policy[row, column] = action
                # Setting the value function for each position as the highest action value (Greedy Policy Update)
                new_value_function[row, column] = max(action_values)
                # Track the largest delta within each iteration of the grid
                delta = max(delta, abs(new_value_function[row, column] - value_function[row, column]))
        
        # Update the value function with the new values
        value_function[:] = new_value_function
        value_log.append(np.copy(value_function))
        
        # Break once the delta is below the threshold
        if delta < theta:
            break
        
        iteration += 1

# Print the optimal policy from the optimal (final) value function
def print_optimal_policy():
    global policy_grid
    # Extract policy grid from best actions
    for row in range(maze.shape[0]):
        for column in range(maze.shape[1]):
            if greedy_policy[row, column] is not None:
                policy_grid[row, column] = action_symbols[greedy_policy[row, column]]
    print("\nOptimal Policy Grid:")
    # Spacing between each element in the row
    for row in policy_grid:
        print(" ".join(row))

# Plots a few snapshots of how the value function evolves
def plot_value_function_evolution():
    num_iterations = len(value_log)
    plt.figure(figsize=(8, 4))
    # Set of plots (start, early, mid, end)
    plot_indices = [0, num_iterations // 4, num_iterations // 2, num_iterations - 1]
    for i, idx in enumerate(plot_indices):
        # Subplot for each index
        plt.subplot(1, len(plot_indices), i + 1)
        # Colour heatmap for the value function, with no smoothing of pixel colours between squares
        plt.imshow(value_log[idx], cmap="coolwarm", interpolation="nearest")
        plt.colorbar()
        plt.title(f"Iteration {idx}")
    
    plt.suptitle("Stages of the Value Function iterations")
    plt.show()

# Run value iteration, print optimal policy, and plot value function convergence
value_iteration()
print_optimal_policy()
plot_value_function_evolution()
