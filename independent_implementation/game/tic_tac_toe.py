from abc import ABC, abstractmethod
from game_interface import Game
import numpy as np

"""
Below is a complete Python implementation of TicTacToe that conforms to the generic Game interface. 
This implementation uses a state representation that is a tuple of (board, current_player), 
where the board is a 3×3 list‐of‐lists (with 0 for empty, 1 for player 1, and –1 for player 2), 
and current_player is 1 (player 1) or –1 (player 2).

For the optional symmetry method, we use NumPy to generate the eight board symmetries 
(four rotations, each with a horizontal flip variant) and apply the same transformation 
to a flat policy vector (assumed to be of length 9 in row‐major order).
"""

class TicTacToe(Game):
    def __init__(self):
        self.size = 3  # board dimensions: 3x3

    def get_initial_state(self):
        # The board is a 3x3 list of 0's, and player 1 starts.
        board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        return (board, 1)

    def get_current_player(self, state):
        # State is (board, current_player)
        _, current_player = state
        return current_player

    def get_legal_moves(self, state):
        board, _ = state
        moves = []
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == 0:
                    moves.append((i, j))
        return moves

    def get_next_state(self, state, move):
        board, player = state
        i, j = move
        if board[i][j] != 0:
            raise ValueError("Invalid move: position already occupied.")
        # Create a deep copy of the board
        new_board = [row[:] for row in board]
        new_board[i][j] = player
        # Switch player
        return (new_board, -player)

    def is_terminal(self, state):
        # Terminal if there is a winner or if the board is full.
        if self.get_winner(state) is not None:
            return True
        board, _ = state
        # Check if any empty cell remains.
        for row in board:
            if 0 in row:
                return False
        return True

    def get_winner(self, state):
        board, _ = state

        # Check rows for a win.
        for i in range(self.size):
            if board[i][0] != 0 and board[i][0] == board[i][1] == board[i][2]:
                return board[i][0]

        # Check columns for a win.
        for j in range(self.size):
            if board[0][j] != 0 and board[0][j] == board[1][j] == board[2][j]:
                return board[0][j]

        # Check diagonals.
        if board[0][0] != 0 and board[0][0] == board[1][1] == board[2][2]:
            return board[0][0]
        if board[0][2] != 0 and board[0][2] == board[1][1] == board[2][0]:
            return board[0][2]

        # If the board is full and no winner, it's a draw.
        if all(cell != 0 for row in board for cell in row):
            return 0

        # Otherwise, the game is not over.
        return None

    def serialize_state(self, state):
        board, player = state
        # Convert board rows to tuples and then wrap in a tuple; include current player.
        board_tuple = tuple(tuple(row) for row in board)
        return (board_tuple, player)

    def get_symmetries(self, state, pi):
        """
        Generate all 8 symmetric transformations of the TicTacToe board.
        Here, pi is assumed to be a flat list of 9 move probabilities corresponding
        to positions in row-major order.
        Returns a list of tuples: (symmetric_state, symmetric_pi)
        """
        board, player = state
        board_np = np.array(board)
        pi_np = np.array(pi).reshape(self.size, self.size)
        symmetries = []

        # Apply 4 rotations (0, 90, 180, 270 degrees)
        for r in range(4):
            rotated_board = np.rot90(board_np, r)
            rotated_pi = np.rot90(pi_np, r)
            # Identity symmetry for this rotation.
            sym_state = (rotated_board.tolist(), player)
            sym_pi = rotated_pi.flatten().tolist()
            symmetries.append((sym_state, sym_pi))
            # Also include the reflection of the rotated board (flip horizontally).
            flipped_board = np.fliplr(rotated_board)
            flipped_pi = np.fliplr(rotated_pi)
            sym_state_flipped = (flipped_board.tolist(), player)
            sym_pi_flipped = flipped_pi.flatten().tolist()
            symmetries.append((sym_state_flipped, sym_pi_flipped))

        return symmetries

    def render(self, state):
        """
        Return a pretty-printed string representation of the TicTacToe board.
        We'll represent player 1 as 'X', player -1 as 'O', and empty cells as ' '.
        """
        board, current_player = state
        symbols = {1: "X", -1: "O", 0: " "}
        lines = []
        for row in board:
            # Convert each cell to its symbol.
            line = " | ".join(symbols[cell] for cell in row)
            lines.append(line)
        separator = "\n" + "-" * (self.size * 4 - 3) + "\n"
        return separator.join(lines)

# --- Example usage ---
if __name__ == "__main__":
    game = TicTacToe()
    state = game.get_initial_state()
    print("Initial state:")
    print(game.render(state))
    # Make a move and show updated board.
    state = game.get_next_state(state, (0, 0))
    print("\nAfter move (0,0):")
    print(game.render(state))