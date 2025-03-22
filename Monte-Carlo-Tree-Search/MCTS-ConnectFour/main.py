from connect_four import ConnectFour
from node import Node
from mcts import MCTS
from mcts_player import MCTSPlayer
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def display_board(board):
    # Display the ConnectFour board in a readable format
    print("\n")
    
    # Print column numbers
    print(" ", end="")
    for i in range(board.shape[1]):
        print(f" {i}", end="")
    print("\n")
    
    for row in range(board.shape[0]):
        print("|", end="")
        for col in range(board.shape[1]):
            piece = board[row, col]
            # Empty space
            if piece == 0:
                print(" ·", end="")
            # Human player piece
            elif piece == 1:
                print(" X", end="")
            # AI player piece
            else:
                print(" O", end="")
        print(" |")
    
    # Print bottom border
    print("+", end="")
    for i in range(board.shape[1]):
        print("--", end="")
    print("-+")
    print("\n")

# Determines if the game has ended by checking both players' terminal states
def check_game_over(node):
    # Returns True if the game is over (win or draw), False otherwise
    result_human = node.state.is_terminal(1)
    result_ai = node.state.is_terminal(2)
    
    if result_human == node.state.win_score:
        print("Game Over! X (Human) wins!")
        return True
    if result_ai == node.state.win_score:
        print("Game Over! O (AI) wins!")
        return True
    if result_human == node.state.draw_score or result_ai == node.state.draw_score:
        print("Game Over! It's a draw!")
        return True
    
    return False

def main():
    # Initialize game components
    game = ConnectFour()
    mcts = MCTS()
    ai_player = MCTSPlayer(mcts)
    current_node = Node(game, None, None)
    
    print("Welcome to Connect Four!")
    print("You are X, AI is O. You go first!")
    print("Enter a column number (0-6) to drop your piece.")

    while True:
        # Always show the current board before a move
        display_board(current_node.state.board)

        # Check if the game ended on the previous move
        if check_game_over(current_node):
            break

        # Human player's turn
        if current_node.state.current_player == 1:
            move = ai_player.get_move_from_human(current_node)
            if move is None:
                print("Game ended by player.")
                sys.exit(0)
            game.apply_move(move)
            current_node = Node(game, current_node, move)

        # AI player's turn
        else:
            print("AI is thinking...")
            best_move = ai_player.get_move(current_node)
            
            # Look up the child node for that best_move to get its UCB stats
            child_node = current_node.child_nodes.get(best_move, None)
            if child_node is not None:
                ucb_value = mcts.ucb(child_node)
                print(f"UCB Value for AI move {best_move}: {ucb_value}")
                print(f"Number of Visits for AI move {best_move}: {child_node.num_visits}")
            else:
                # Fallback if for some reason we can't find the child node
                print(f"No child node found for move {best_move}. (UCB not available)")

            game.apply_move(best_move)
            current_node = Node(game, current_node, best_move)
            print(f"AI placed piece in column {best_move}")


if __name__ == "__main__":
    main()
