from abc import ABC, abstractmethod

class Game(ABC):
    @abstractmethod
    def get_initial_state(self):
        """
        Return the initial game state.
        The state can be in any form (list, numpy array, custom object) as long as it is compatible with the other methods.
        """
        pass

    @abstractmethod
    def get_current_player(self, state):
        """
        Given a state, return the current player (e.g., 1 or -1, or an identifier).
        This is useful when evaluating rewards and for the learning algorithm to know which perspective to use.
        """
        pass

    @abstractmethod
    def get_legal_moves(self, state):
        """
        Given a state, return a list (or other iterable) of legal moves.
        The move representation should be compatible with how the game transitions are handled.
        """
        pass

    @abstractmethod
    def get_next_state(self, state, move):
        """
        Given a state and a legal move, return the new state after applying that move.
        Note that if your states are mutable, you might want to return a deep copy or use a clone method.
        """
        pass

    @abstractmethod
    def is_terminal(self, state):
        """
        Return True if the state is terminal (i.e. the game is over), False otherwise.
        """
        pass

    @abstractmethod
    def get_winner(self, state):
        """
        If the state is terminal, return the winner (for example, 1 for player 1, -1 for player 2,
        or 0 for a draw). If the game is not over, this could return None.
        """
        pass

    @abstractmethod
    def serialize_state(self, state):
        """
        Return a canonical or hashable representation of the state.
        This can be useful for caching, logging, or storing training examples.
        """
        pass

    @abstractmethod
    def get_symmetries(self, state, pi):
        """
        (Optional) Given a state and a move probability vector pi (from MCTS or network output),
        return a list of (state, pi) pairs that are symmetric equivalents.
        This can be used to augment the training data in AlphaZero-like training.
        """
        pass

    @abstractmethod
    def render(self, state):
        """
        Return a human-readable string representation of the state.
        This can be used for debugging, logging, or even displaying the game.
        """
        pass
