# TD(0) Maze-Grid solver - deterministic MDP
import numpy as np
import matplotlib.pyplot as plt
import random

# Size of maze
rows = 12
columns = 12

# 12x12 maze where 1 = wall, 0 = path
maze = np.array([
    [1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
    [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
])
#print(maze)

# All possible moves (down, right, up, left)
all_moves = [
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1]
]

# Map from moves to arrows
moves_to_arrows = {
    (1, 0): '↓',
    (0, 1): '→',
    (-1, 0): '↑',
    (0, -1): '←'
}

# Start point
start_point = [10, 0]

# Move reward (penalty)
step_reward = -1

# Danger point
danger_point = [0, 7]
danger_reward = -5

# Teasure point
treasure_point = [8, 11]
treasure_reward = 10

def initial_plot():
    # Plot figure
    plt.figure(figsize=(6, 6))
    # Use greyscale
    plt.imshow(maze, cmap="Greys")
    # Add a title
    plt.title("Maze (12x12)")
    # Get current axis
    ax = plt.gca()
    # Add minor grid lines horizontally and vertically
    ax.set_xticks(np.arange(-0.5, columns, 1), minor = True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor = True)
    # Turn on grid for minor ticks and format
    ax.grid(which='minor', color = 'black', linestyle = '-', linewidth = 1)
    # Remove both axis ticks (hide tick marks, and numbers)
    plt.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    # Add a black spot for the start point using a scatter plot
    plt.scatter(start_point[1], start_point[0], c = 'black', s = 100)
    # Add a red spot for the danger point using a scatter plot
    plt.scatter(danger_point[1], danger_point[0], c = 'red', s = 100)
    # Add a green spot for the treasure point using a scatter plot
    plt.scatter(treasure_point[1], treasure_point[0], c = 'green', s = 100)
    # Show
    plt.show()
    return
#initial_plot()

# Reward matrix
rewards = -1*np.ones((rows, columns))
rewards[danger_point[0], danger_point[1]] = danger_reward
rewards[treasure_point[0], treasure_point[1]] = treasure_reward
print(rewards)

# Apply move to a state
def apply_move(state, move):
    vertical = state[0] + move[0]
    horizontal = state[1] + move[1]
    return [vertical, horizontal]

# Check if state is within bounds
def within_bounds(state):
    # Check if state is within grid
    if not(0 <= state[0] < 12) or not(0 <= state[1] < 12):
        return False
    # Check if state is on path (not wall)
    if maze[state[0], state[1]] == 1:
        return False
    # Return true if both conditions are not true
    return True

# Get valid moves
def get_valid_moves(state):
    valid_moves = []
    for i in range(4):
        if within_bounds(apply_move(state, all_moves[i])) == True:
            valid_moves.append(all_moves[i])
    return valid_moves
#print(get_valid_moves(start_point))

# Random policy
def random_policy(state):
    moves = get_valid_moves(state)
    move = random.choice(moves)
    next_state = apply_move(state, move)
    #print("At position: ", state, " & Choosing move: ", move, " & Going to position: ", next_state)
    return next_state
#print(random_policy(start_point))

# Initialise a value matrix
values = np.zeros((rows, columns))
#print(values)

# Epsilon-greedy policy
initial_epsilon = 1.00
def episilon_greedy_policy(state, epsilon):
    greedy = epsilon > random.uniform(0,1)
    if greedy == True:
        moves = get_valid_moves(state)
        greediest_move = None
        greediest_move_value = float('-inf')
        for move in moves:
            next_state = apply_move(state, move)
            next_state_value = values[next_state[0], next_state[1]]
            if next_state_value > greediest_move_value:
                greediest_move = move
                greediest_move_value = next_state_value
        return apply_move(state, greediest_move)
    else: 
        return random_policy(state)
#print(episilon_greedy_policy(start_point, initial_epsilon))

# Length of a trajectory for an episode
len_trajectory = 200

# Set max learning rate
max_alpha = 1.00

# Set discount rate
discount_rate = 1.00

# Number of episodes 
num_episodes = 1000

# Initialise a visit count
state_visit_count = np.zeros((rows, columns))

# Return a trajectory for an episode
def run_episode(start, length, epsilon):
    state = start
    for _ in range(length):
        state_visit_count[state[0],state[1]] += 1
        alpha = min(max_alpha, 1 / state_visit_count[state[0],state[1]])
        value_old_state = values[state[0], state[1]]
        #print(value_old_state)
        new_state = episilon_greedy_policy(state, epsilon)
        #print(state)
        value_new_state = values[new_state[0], new_state[1]]
        #print(value_new_state)
        reward_new_state = rewards[new_state[0], new_state[1]]
        #print(reward_new_state)
        td_target = reward_new_state + discount_rate*value_new_state
        values[state[0],state[1]] += alpha*(td_target-value_old_state)
        #print(values)
        #print("")
        if new_state == treasure_point:
            break
        state = new_state
    return values
#print(run_episode(start_point, len_trajectory))

# Run the TD0 process
for i in range(num_episodes):
    epsilon = initial_epsilon / (i+1)
    if i % 500 == 0:
        print("Episode: ", i)
    run_episode(start_point, len_trajectory, epsilon)

# Print the values matrix and state visit count
print(np.round(values, 1))
print(state_visit_count)

# Create the optimal policy from the values
def run_optimal_policy():
    # Initialise an optimal policy matrix
    optimal_policy = [[] for _ in range(rows)]
    for i in range(rows):
        for j in range(columns):
            # No policy if wall
            if maze[i, j] == 1:
                optimal_policy[i].append(0)
            else: 
                # Get valid moves
                valid_moves = get_valid_moves([i,j])
                best_move = None
                best_move_value = float('-inf')
                # Find the best move
                for move in valid_moves:
                    move_position = apply_move([i, j], move)
                    move_value = values[move_position[0], move_position[1]]
                    if move_value > best_move_value:
                        best_move_value = move_value
                        best_move = move
                # Set the best move as the optimal policy
                optimal_policy[i].append(best_move)
    return optimal_policy
run_optimal_policy()
#print(run_optimal_policy())

# Convert optimal policy to arrows
def graphical_optimal_policy(optimal_policy):
    optimal_policy_arrows = [[] for _ in range(rows)]
    for i in range(rows): 
        for j in range(columns):
            # No policy if wall
            if maze[i, j] == 1:
                optimal_policy_arrows[i].append('·')
            else:
                move = optimal_policy[i][j]
                if tuple(move) in moves_to_arrows:
                    optimal_policy_arrows[i].append(moves_to_arrows[tuple(move)])
    return optimal_policy_arrows
#graphical_optimal_policy(run_optimal_policy())

def print_graphical_optimal_policy(optimal_policy_arrows):
    for row in optimal_policy_arrows:
        print(' '.join(row))
print_graphical_optimal_policy(graphical_optimal_policy(run_optimal_policy()))
