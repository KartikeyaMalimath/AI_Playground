import logging

import numpy as np


class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None
        return self.board

    def available_actions(self):
        available_actions = [action + 1 for action in range(9) if self.board[action // 3, action % 3] == 0]
        return available_actions

    def step(self, action, player):
        if self.done:
            raise ValueError("Game is over")
        row, col = divmod(action - 1, 3)
        if self.board[row, col] != 0:
            raise ValueError("Invalid action")
        self.board[row, col] = player  # Corrected line
        self.done, self.winner = self.check_winner()
        return self.board, self.done, self.winner

    def check_winner(self):
        for player in [1, 2]:
            if np.any(np.all(self.board == player, axis=0)) or \
                    np.any(np.all(self.board == player, axis=1)) or \
                    np.all(np.diag(self.board) == player) or \
                    np.all(np.diag(np.fliplr(self.board)) == player):
                return True, player
        if np.all(self.board != 0):
            return True, 0
        return False, None

    def render(self):
        symbols = {0: '.', 1: 'X', 2: 'O'}
        for row in self.board:
            print(' '.join(symbols[cell] for cell in row))
        print()

    @staticmethod
    def human_move(env):
        while True:
            move = input("Enter your move (1-9): ")
            try:
                move = int(move)
                if 1 <= move <= 9:
                    if move in env.available_actions():
                        return move
            except Exception as e:
                logging.error(f"Error: {e}")
            print("Invalid move. Try again.")
