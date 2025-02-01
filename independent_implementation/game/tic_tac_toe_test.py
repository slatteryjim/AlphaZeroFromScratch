import unittest
from tic_tac_toe import TicTacToe

class TestTicTacToe(unittest.TestCase):
    def test_board_as_string(self):
        game = TicTacToe()
        self.assertEqual(game.board_as_string(),
            "000\n"+
            "000\n"+
            "000\n")
        
    def test_available_moves(self):
        game = TicTacToe()
        self.assertEqual(game.available_moves(), [0,1,2,3,4,5,6,7,8])

    def test_move_index_to_position(self):
        game = TicTacToe()
        self.assertEqual(game.move_index_to_position(0), (0,0))
        self.assertEqual(game.move_index_to_position(1), (0,1))
        self.assertEqual(game.move_index_to_position(2), (0,2))
        self.assertEqual(game.move_index_to_position(3), (1,0))
        self.assertEqual(game.move_index_to_position(4), (1,1))
        self.assertEqual(game.move_index_to_position(5), (1,2))
        self.assertEqual(game.move_index_to_position(6), (2,0))
        self.assertEqual(game.move_index_to_position(7), (2,1))
        self.assertEqual(game.move_index_to_position(8), (2,2))
        
        with self.assertRaises(IndexError) as cm:
            game.move_index_to_position(-1)
        self.assertEqual(str(cm.exception), "Move index cannot be less than zero")

        with self.assertRaises(IndexError) as cm:
            game.move_index_to_position(9)
        self.assertEqual(str(cm.exception), "Move index out of range")

    def test_play(self):
        game = TicTacToe()
        self.assertEqual(game.board_as_string(), "000\n000\n000\n")
        self.assertEqual(game.available_moves(), [0,1,2,3,4,5,6,7,8])

        # play a move
        game.play(1, 2)
        self.assertEqual(game.board_as_string(), "001\n000\n000\n")
        self.assertEqual(game.available_moves(), [0,1,3,4,5,6,7,8])

        # play a move again fails
        with self.assertRaises(IndexError) as cm:
            game.play(1, 2)
        self.assertEqual(str(cm.exception), "play: invalid move_index 2 (0,2) already has value: 1")

        # play a move
        game.play(-1, 7)
        self.assertEqual(game.board_as_string(), "001\n000\n0-10\n")
        self.assertEqual(game.available_moves(), [0,1,3,4,5,6,8])

    def test_is_terminal(self):
        game = TicTacToe()
        self.assertFalse(game.is_terminal())

        # check rows
        game.play(1, 0)
        game.play(1, 1)
        self.assertFalse(game.is_terminal())
        game.play(1, 2)
        self.assertTrue(game.is_terminal())

        # check columns
        game = TicTacToe()
        game.play(-1, 0)
        game.play(-1, 3)
        self.assertFalse(game.is_terminal())
        game.play(-1, 6)
        self.assertTrue(game.is_terminal())

        # check diagonals
        game = TicTacToe()
        game.play(1, 0)
        game.play(1, 4)
        self.assertFalse(game.is_terminal())
        game.play(1, 8)
        self.assertTrue(game.is_terminal())

        game = TicTacToe()
        game.play(-1, 2)
        game.play(-1, 4)
        self.assertFalse(game.is_terminal())
        game.play(-1, 6)
        self.assertTrue(game.is_terminal())

        game = TicTacToe()
        game.play(1, 3)
        game.play(-1, 0)
        game.play(1, 4)
        game.play(-1, 1)
        self.assertFalse(game.is_terminal())
        game.play(1, 5)
        self.assertTrue(game.is_terminal())
        
if __name__ == '__main__':
    unittest.main()