# A node in the MCTS tree
class Node:
    # Initialise a new node for the MCTS tree
    def __init__(self, state, parent_node, action_taken):
        self.state = state
        self.current_player = state.current_player
        self.parent_node = parent_node
        self.action_taken = action_taken
        self.num_visits = 0
        self.num_wins = 0
        self.child_nodes = {}
        self.untried_moves = state.get_valid_actions()

    # Expand this node by applying an untried move and creating a new child node
    def expand(self, action):
        self.untried_moves.remove(action)
        child_state = self.state.copy_state()
        child_state.apply_move(action)
        child_node = Node(child_state, self, action)
        self.child_nodes[action] = child_node
        return child_node

    # Returns the action corresponding to the child with the index highest visit count
    def get_best_actions(self, index):
        if not self.child_nodes:
            # Return None if there are no child nodes
            return None

        # Create a list of actions and their respective visit counts
        action_visits = [(action, child.num_visits) for action, child in self.child_nodes.items()]
        
        # Sort actions by num_visits in descending order
        action_visits.sort(key=lambda x: x[1], reverse=True)
        
        # Return the action corresponding to the provided index
        best_actions = [action for action, visits in action_visits]

        # Check if the index is valid
        if index < 0 or index >= len(best_actions):
            raise IndexError("Index out of range for best actions.")
        
        return best_actions[index]

    
    # Returns the average win rate for this node
    def get_win_rate(self):
        if self.num_visits == 0:
            return 0
        return self.num_wins / self.num_visits
