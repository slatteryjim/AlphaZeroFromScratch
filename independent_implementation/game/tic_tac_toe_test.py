import unittest
from tic_tac_toe import TicTacToe

class TestTicTacToe(unittest.TestCase):
    def setUp(self):
        self.game = TicTacToe()
        self.initial_state = self.game.get_initial_state()

    def test_initial_state(self):
        """Test that the initial state is a 3x3 empty board with player 1 to move."""
        board, player = self.initial_state
        self.assertEqual(player, 1)
        self.assertEqual(len(board), 3)
        for row in board:
            self.assertEqual(row, [0, 0, 0])

    def test_get_legal_moves_initial(self):
        """Test that all 9 moves are legal in the initial state."""
        legal_moves = self.game.get_legal_moves(self.initial_state)
        expected_moves = [(i, j) for i in range(3) for j in range(3)]
        self.assertEqual(len(legal_moves), 9)
        self.assertCountEqual(legal_moves, expected_moves)

    def test_get_next_state_and_current_player(self):
        """Test that making a move updates the board and toggles the player."""
        move = (0, 0)
        next_state = self.game.get_next_state(self.initial_state, move)
        board, player = next_state
        # Check that the move was applied correctly.
        self.assertEqual(board[0][0], 1)
        # Player should have toggled from 1 to -1.
        self.assertEqual(player, -1)
        # The legal moves should no longer include the taken move.
        legal_moves = self.game.get_legal_moves(next_state)
        self.assertEqual(len(legal_moves), 8)
        self.assertNotIn(move, legal_moves)

    def test_invalid_move(self):
        """Test that attempting to play on an occupied cell raises an error."""
        move = (0, 0)
        state_after_move = self.game.get_next_state(self.initial_state, move)
        # Attempting to play the same move on the new state should raise a ValueError.
        with self.assertRaises(ValueError):
            self.game.get_next_state(state_after_move, move)

    def test_is_terminal_and_get_winner_row(self):
        """Test a terminal state where player 1 wins by completing a row."""
        board = [
            [1, 1, 1],
            [0, -1, 0],
            [0, 0, -1]
        ]
        state = (board, -1)
        self.assertTrue(self.game.is_terminal(state))
        self.assertEqual(self.game.get_winner(state), 1)

    def test_is_terminal_and_get_winner_column(self):
        """Test a terminal state where player -1 wins by completing a column."""
        board = [
            [1, -1, 0],
            [0, -1, 1],
            [0, -1, 0]
        ]
        state = (board, 1)
        self.assertTrue(self.game.is_terminal(state))
        self.assertEqual(self.game.get_winner(state), -1)

    def test_is_terminal_and_get_winner_diagonal(self):
        """Test a terminal state where player 1 wins via a diagonal."""
        board = [
            [1, -1, 0],
            [0, 1, -1],
            [0, 0, 1]
        ]
        state = (board, -1)
        self.assertTrue(self.game.is_terminal(state))
        self.assertEqual(self.game.get_winner(state), 1)

    def test_draw(self):
        """Test a terminal draw state (full board with no winner)."""
        board = [
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, -1]
        ]
        state = (board, 1)
        self.assertTrue(self.game.is_terminal(state))
        self.assertEqual(self.game.get_winner(state), 0)

    def test_serialize_state(self):
        """Test that the state serialization produces a hashable and consistent representation."""
        state = self.initial_state
        serialized = self.game.serialize_state(state)
        self.assertIsInstance(serialized, tuple)
        # Calling serialization twice should produce the same result.
        serialized2 = self.game.serialize_state(state)
        self.assertEqual(serialized, serialized2)

    def test_get_symmetries(self):
        """Test that the symmetry function returns 8 transformed versions with proper shapes."""
        # Use a dummy probability vector for 9 moves.
        dummy_pi = [i / 9.0 for i in range(9)]
        symmetries = self.game.get_symmetries(self.initial_state, dummy_pi)
        self.assertEqual(len(symmetries), 8)
        for sym_state, sym_pi in symmetries:
            board, player = sym_state
            self.assertEqual(len(board), 3)
            for row in board:
                self.assertEqual(len(row), 3)
            self.assertEqual(len(sym_pi), 9)

if __name__ == "__main__":
    unittest.main()
