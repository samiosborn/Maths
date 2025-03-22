import numpy as np
import copy

# ConnectFour class
class ConnectFour:
    # Number of columns of board
    num_columns = 7
    # Number of rows of board
    num_rows = 6
    # Fixed win definition for ConnectFour
    winning_length = 4
    # Columns which a disc can be placed
    action_set = [0, 1, 2, 3, 4, 5, 6]
    # Beginning player
    beginning_player = 2
    # Score of 1 if player wins
    win_score = 1
    # Score of 0.5 if game is a draw
    draw_score = 0.5
    # Score if game is not terminal
    not_win_score = 0

    # After running MCTS iterations, select the best move from the root node   
    def __init__(self):
        self.board = np.zeros((ConnectFour.num_rows, ConnectFour.num_columns))
        self.current_player = ConnectFour.beginning_player

    # Apply a move (column index) to the board
    def apply_move(self, action):
        # Apply a move (column index) to the board
        for i in range(ConnectFour.num_rows - 1, -1, -1):
            if self.board[i, action] == 0:
                self.board[i, action] = self.current_player
                # Switch player (1 becomes 2, 2 becomes 1)
                self.current_player = 3 - self.current_player
                return
 
    # Returns a list of valid column indices where a move can be made.
    def get_valid_actions(self):
        # Returns a list of valid column indices where a move can be made
        valid_actions = []
        for i in range(ConnectFour.num_columns):
            if self.board[0, i] == 0:
                valid_actions.append(i)
        return valid_actions

    # Checks for a win, loss, or draw
    def is_terminal(self, player):
        # Checks for a win, loss, or draw
        # Check horizontally
        for i in range(ConnectFour.num_rows):
            count = 0
            for j in range(ConnectFour.num_columns):
                if self.board[i, j] == player:
                    count += 1
                    if count == ConnectFour.winning_length:
                        return ConnectFour.win_score
                else:
                    count = 0

        # Check vertically
        for j in range(ConnectFour.num_columns):
            count = 0
            for i in range(ConnectFour.num_rows):
                if self.board[i, j] == player:
                    count += 1
                    if count == ConnectFour.winning_length:
                        return ConnectFour.win_score
                else:
                    count = 0

        # Check diagonally (down-right)
        for i in range(ConnectFour.num_rows - ConnectFour.winning_length + 1):
            for j in range(ConnectFour.num_columns - ConnectFour.winning_length + 1):
                if all(self.board[i+k, j+k] == player for k in range(ConnectFour.winning_length)):
                    return ConnectFour.win_score

        # Check diagonally (down-left)
        for i in range(ConnectFour.num_rows - ConnectFour.winning_length + 1):
            for j in range(ConnectFour.winning_length - 1, ConnectFour.num_columns):
                if all(self.board[i+k, j-k] == player for k in range(ConnectFour.winning_length)):
                    return ConnectFour.win_score

        # Check for draw
        if not self.get_valid_actions():
            return ConnectFour.draw_score

        return ConnectFour.not_win_score

    # Returns winner of the game, if any
    def get_winner(self):
        if self.is_terminal(1) == self.win_score:
            return 1
        elif self.is_terminal(2) == self.win_score:
            return 2
        else:
            return None

    # Creates a deep copy of the current state (board and current player)
    def copy_state(self):
        # Creates a deep copy of the current state (board and current player)
        new_state = ConnectFour()
        new_state.board = copy.deepcopy(self.board)
        new_state.current_player = self.current_player
        return new_state
