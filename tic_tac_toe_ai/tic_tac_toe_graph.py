import matplotlib.pyplot as plt
import numpy as np


class TicTacToeGraphPlotter:
    def __init__(self):
        self.player1_actions = []
        self.player2_actions = []
        self.outcomes = []

    def add_game(self, player1_actions, player2_actions, outcome):
        self.player1_actions.append(player1_actions)
        self.player2_actions.append(player2_actions)
        self.outcomes.append(outcome)

    def plot_predictions(self):
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        self.plot_graph(axs[0], self.player1_actions, 'Player 1 Moves')
        self.plot_graph(axs[1], self.player2_actions, 'Player 2 Moves')
        plt.tight_layout()
        plt.show()

    def plot_graph(self, ax, actions, title):
        ax.set_title(title)
        ax.set_xlabel('Game')
        ax.set_ylabel('Move')
        ax.set_xticks(np.arange(len(actions) + 1))
        ax.set_xticklabels([f'Game {i}' for i in range(len(actions) + 1)])
        ax.set_yticks(np.arange(1, 10))
        ax.grid(True)

        for i, game_actions in enumerate(actions):
            color = 'gray'
            outcome = self.outcomes[i]
            if outcome[0] == 1:
                color = 'green'
            elif outcome[1] == 1:
                color = 'red'
            elif outcome[0] == 0 and outcome[1] == 0:
                color = 'yellow'
            for move in game_actions:
                ax.add_patch(plt.Rectangle((i, move - 0.4), 0.4, 0.8, color=color))
