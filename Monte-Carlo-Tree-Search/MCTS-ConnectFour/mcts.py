import math
import random
import logging

# Monte Carlo Tree Search (MCTS) for ConnectFour
class MCTS:
    # Initialise the MCTS algorithm
    num_iterations = 500
    max_selection_steps = 100000
    explore_parameter = math.sqrt(2)
    max_simulation_steps = 100000

    # Compute the Upper Confidence Bound (UCB) for a node
    def ucb(self, node):
        # If the node has not been visited, return infinity to ensure it is selected
        if node.num_visits == 0 or node.parent_node is None or node.parent_node.num_visits == 0:
            return float('inf')
        # Balances exploitation and exploration
        exploitation = node.num_wins / node.num_visits
        exploration = (self.explore_parameter *
                       math.sqrt(math.log(node.parent_node.num_visits) / node.num_visits))
        return exploitation + exploration

    # Selection phase - descend the tree, always picking the child with the highest UCB
    def selection(self, node):
        # Return None if the node has no children
        if not node.child_nodes:
            return None

        # Log child UCBs for each child node for debugging
        ucb_values = {action: self.ucb(child) for action, child in node.child_nodes.items()}
        logging.debug(f"[SELECTION] Child UCBs for node (action_taken={node.action_taken}): {ucb_values}")

        # Pick the child with the highest UCB
        best_child = max(node.child_nodes.values(), key=self.ucb)
        logging.debug(f"[SELECTION] Selected action = {best_child.action_taken} with UCB = {self.ucb(best_child)}")
        return best_child

    # Expansion phase - if the node has untried moves, pick one at random, expand the tree
    def expansion(self, node):
        # If the node has no untried moves, return None
        if not node.untried_moves:
            return None
        # Pick a random untried move
        expansion_action = random.choice(node.untried_moves)
        logging.debug(f"[EXPANSION] Expanding node with new action = {expansion_action}")
        # Expand the tree with the new action
        new_node = node.expand(expansion_action)
        return new_node

    # Simulation phase - randomly play until we hit a terminal state or max steps (rollout)
    def simulation(self, node):
        # Randomly play until we hit a terminal state or max steps
        current_node = node
        steps = 0
        while steps < self.max_simulation_steps:
            # Get the last mover
            last_mover = 3 - current_node.state.current_player
            # If the node is a terminal state, break
            if current_node.state.is_terminal(last_mover) != current_node.state.not_win_score:
                break
            # Get the valid moves, if there are no valid moves, break
            valid_moves = current_node.state.get_valid_actions()
            if not valid_moves:
                break
            # Pick a random valid move
            move = random.choice(valid_moves)
            # Expand the tree with the new action
            current_node = current_node.expand(move)
            steps += 1
        # Return the end or terminal node
        return current_node

    # Backpropagation phase - walk up the tree from the terminal node, updating statistics
    def backpropagation(self, terminal_node):
        # Get the last mover
        last_mover = 3 - terminal_node.current_player
        # Loop until we reach the root node
        while terminal_node is not None:
            # Update the number of visits
            terminal_node.num_visits += 1
            # If the current player is the last mover, increment or decrement the number of wins
            if terminal_node.current_player == last_mover:
                terminal_node.num_wins += 1
            else:
                terminal_node.num_wins -= 1
            # Log the node's statistics
            logging.debug(
                f"[BACKPROP] Node(action_taken={terminal_node.action_taken}), "
                f"num_visits={terminal_node.num_visits}, num_wins={terminal_node.num_wins}"
            )
            # Move up to the parent node
            terminal_node = terminal_node.parent_node

    # Run a single iteration of the full MCTS cycle on the current node
    def run_mcts(self, node):
        # Selection phase - select the best leaf node by UCB
        steps = 0
        while steps < self.max_selection_steps:
            last_mover = 3 - node.state.current_player
            # If node is terminal or untried moves exist, break
            if node.state.is_terminal(last_mover) != node.state.not_win_score or node.untried_moves:
                break
            # Otherwise pick the best child by UCB and keep going down
            selected_child = self.selection(node)
            if selected_child is None:
                break
            # Move down to the selected child
            node = selected_child
            steps += 1

        # Expansion phase - expand the tree if the node has untried moves and is not terminal
        last_mover = 3 - node.state.current_player
        if node.untried_moves and node.state.is_terminal(last_mover) == node.state.not_win_score:
            node = self.expansion(node)

        # Simulation phase - randomly play until we hit a terminal state or max steps (rollout)
        terminal_node = self.simulation(node)

        # Backpropagation phase - walk up the tree from the terminal node, updating statistics
        self.backpropagation(terminal_node)

    # Check for an immediate winning move for the current player
    def immediate_win_move(self, node):
        mover = node.current_player
        valid_moves = node.state.get_valid_actions()
        logging.debug(f"Checking immediate win moves for player {mover}. Valid moves: {valid_moves}")
        logging.debug(f"Board before checking immediate win:\n{node.state.board}")

        # Copies the current state and tries each valid move to see if it's a winning move
        for move in valid_moves:
            temp_state = node.state.copy_state()
            temp_state.apply_move(move)
            if temp_state.is_terminal(mover) == temp_state.win_score:
                logging.debug(f"Winning move found for player {mover} at column {move}")
                return move
        logging.debug(f"No immediate winning move found for player {mover}")
        return None

    # Check for an immediate blocking move for the opponent
    def immediate_block_move(self, node):
        current_player = node.state.current_player
        opponent = 1 if current_player == 2 else 2

        # Copies the current state and gets the valid moves
        original_state = node.state.copy_state()
        valid_moves = original_state.get_valid_actions()

        # Identify the opponent's next-move winning columns
        opponent_winning_moves = []
        for opp_move in valid_moves:
            test_state = original_state.copy_state()
            test_state.current_player = opponent
            test_state.apply_move(opp_move)
            if test_state.is_terminal(opponent) == test_state.win_score:
                opponent_winning_moves.append(opp_move)
        
        # If there are no opponent winning moves, return None
        if not opponent_winning_moves:
            logging.debug(f"No immediate winning move found for opponent {opponent}. No block needed.")
            return None
        
        # Log the opponent's winning moves
        logging.debug(
            f"Opponent {opponent} can immediately win with moves: {opponent_winning_moves}. "
            "Attempting to block at least one..."
        )
        # For each valid AI move, see if it blocks any of those winning columns
        for ai_move in valid_moves:
            simulated_state = original_state.copy_state()
            simulated_state.apply_move(ai_move)  # AI places piece
            # For each opponent winning move, see if it blocks any of those winning columns
            for wmove in opponent_winning_moves:
                if wmove in simulated_state.get_valid_actions():
                    check_state = simulated_state.copy_state()
                    check_state.current_player = opponent
                    check_state.apply_move(wmove)
                    if check_state.get_winner() != opponent:
                        logging.debug(
                            f"Blocking at least one threat: AI move {ai_move} "
                            f"blocks opponent's winning column {wmove}."
                        )
                        return ai_move

        # If no move blocks any of the opponent's winning moves, return None
        logging.debug(f"No single AI move blocks even one of opponent {opponent}'s winning columns.")
        return None

    # Select the best move for the current node
    def select_best_move(self, node):
        # Check immediate winning move
        immediate_win = self.immediate_win_move(node)
        if immediate_win is not None:
            logging.debug(f"Immediate winning move detected: {immediate_win}")
            return immediate_win

        # Check immediate block
        immediate_block = self.immediate_block_move(node)
        if immediate_block is not None:
            logging.debug(f"Immediate blocking move detected: {immediate_block}")
            # Run some MCTS iterations after picking the block
            for _ in range(self.num_iterations):
                self.run_mcts(node)
            return immediate_block

        # Run full MCTS iterations, otherwise
        logging.debug("No forced move found. Running full MCTS iterations.")
        for _ in range(self.num_iterations):
            self.run_mcts(node)

        # After MCTS, pick the child with the highest UCB
        child_ucb_values = {action: self.ucb(child) for action, child in node.child_nodes.items()}
        logging.debug(f"[FINAL] Child UCBs: {child_ucb_values}")

        # Pick the child with the highest UCB
        best_move = max(child_ucb_values, key=child_ucb_values.get)
        logging.debug(f"[FINAL] Best move = {best_move} with UCB = {child_ucb_values[best_move]}")
        return best_move
