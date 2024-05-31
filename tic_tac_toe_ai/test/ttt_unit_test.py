import unittest
from tic_tac_toe_ai.tic_tac_toe_ai import play_game


import unittest


class TestGameModeSelection(unittest.TestCase):

    def test_all(self):
        self.test_game_mode_selection()
        self.test_invalid_moves()
        self.test_winning_game()

    def test_game_mode_selection(self):
        # Test if the game can handle different game modes
        game_modes = [1, 2, 3]
        for mode in game_modes:
            with self.subTest(mode=mode):
                play_game()

    def test_invalid_moves(self):
        # Test if the game can handle invalid moves
        with self.assertRaises(ValueError):
            play_game()
            # Simulate an invalid move by entering a number outside the range of 1-9
            input("Enter your move (1-9): ")

    def test_winning_game(self):
        # Test if the game can handle a winning move
        with self.assertRaises(AssertionError):
            play_game()
            # Simulate a winning game by making moves that lead to a win
            input("Enter your move (1-9): ")
            input("Enter your move (1-9): ")
            input("Enter your move (1-9): ")
            input("Enter your move (1-9): ")
            input("Enter your move (1-9): ")
            input("Enter your move (1-9): ")
            input("Enter your move (1-9): ")
            input("Enter your move (1-9): ")
            input("Enter your move (1-9): ")
            input("Enter your move (1-9): ")

    def test_max_counter_game(self):
        # Test if the game can handle a game with maximum counter
        with self.assertRaises(AssertionError):
            play_game()
            # Simulate a game with maximum counter by entering moves until the counter reaches the maximum value
            for _ in range(10):
                input("Enter your move (1-9): ")


if __name__ == '__main__':
    TestGameModeSelection.test_all()