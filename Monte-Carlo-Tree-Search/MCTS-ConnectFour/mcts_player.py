# MCTSPlayer class for the AI player
class MCTSPlayer:
    # Initialise the MCTSPlayer with an instance of the MCTS algorithm
    def __init__(self, mcts):
        self.mcts = mcts

    # AI move selection
    def get_move(self, node):
        # Takes best move from MCTS algorithm
        move = self.mcts.select_best_move(node)
        print(f"AI selects move: {move}")
        return move

    # Human move selection
    def get_move_from_human(self, node):
        # Prompt the human player for a move and validate the input
        try:
            user_input = input("Enter your move (0-6) or 'q' to quit: ")
            if user_input.lower() in ['q', 'quit']:
                return None
            move = int(user_input)
            valid_moves = node.state.get_valid_actions()
            if move in valid_moves:
                return move
            else:
                print("Invalid move. Try again.")
                return self.get_move_from_human(node)
        except ValueError:
            print("Invalid input. Please enter an integer or 'q' to quit.")
            return self.get_move_from_human(node)
