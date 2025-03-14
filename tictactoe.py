import numpy as np

class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))

    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action is None:
            return False

        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]

        return (
            np.sum(state[row, :]) == player * self.column_count
            or np.sum(state[:, column]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
        )

    def get_value_and_terminated(self, state, action):
        """
        Determines the value and termination status of a game state after a given action.

        Args:
            state (np.ndarray): The current state of the game board.
            action (int): The action to be evaluated.

        Returns:
            tuple: A tuple containing:
                - int: The value of the state (1 if the action results in a win, 0 otherwise).
                - bool: True if the game is terminated (either win or draw), False otherwise.
        """
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        """
        Encodes the 2D game state into a binary representation suitable for neural network input:
        a 3D binary tensor with shape (3, height, width).
        """
        return np.stack((
            state == -1, # all squares where player -1 has played
            state == 0,  # all empty squares
            state == 1   # all squares where player 1 has played
        )).astype(np.float32)