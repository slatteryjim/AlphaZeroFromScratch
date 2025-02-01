
class TicTacToe:

    def __init__(self):
        """creates a board, hard-coded as 3x3"""
        self.dim = 3
        self.board = [[0 for _ in range(self.dim)] for _ in range(self.dim)]
        self.current_player = 1
    
    def board_as_string(self):
        """return a printable string representation of the board"""
        res = ""
        for row in self.board:
            for cell in row:
                res += str(cell)
            res += "\n"
        return res
    
    def available_moves(self):
        res = []
        for i, row in enumerate(self.board):
            for j, cell in enumerate(row):
                if cell == 0:
                    res.append((i*len(row))+j)
        return res
    
    def move_index_to_position(self, move_index):
        if move_index < 0:
            raise IndexError("Move index cannot be less than zero")
        if move_index >= (self.dim * self.dim):
            raise IndexError("Move index out of range")
        x = move_index // self.dim
        y = move_index % self.dim
        return x, y

    def play(self, player, move_index):
        x, y = self.move_index_to_position(move_index)
        if self.board[x][y] != 0:
            raise IndexError(f"play: invalid move_index {move_index} ({x},{y}) already has value: {self.board[x][y]}")
        self.board[x][y] = player

    def is_terminal(self):
        # check rows
        for row in self.board:
            if row[0] == row[1] == row[2] != 0:
                return True
        # check columns
        for col in range(self.dim):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != 0:
                return True
        # check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            return True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            return True
        
        return False