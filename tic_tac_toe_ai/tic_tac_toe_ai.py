import os
import tensorflow as tf
from tic_tac_toe_ai.tic_tac_toe_game import TicTacToe
from tic_tac_toe_ai.tic_tac_toe_graph import TicTacToeGraphPlotter
from tic_tac_toe_ai.ttt_neural_network import create_model, DQNAgent
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_model(model, filename="tic_tac_toe_ai/tic_tac_toe_model.keras"):
    model.save(filename)
    logger.info(f"Model saved to {filename}")


def load_model(filename="tic_tac_toe_ai/tic_tac_toe_model.keras"):
    if os.path.exists(filename):
        model = tf.keras.models.load_model(filename)
        logger.info(f"Model loaded from {filename}")
        return model
    return create_model()


def play_game(agent, players, episodes):
    wanna_play_game = True
    # Example usage:
    plotter = TicTacToeGraphPlotter()

    while wanna_play_game:
        game_over = False
        env = TicTacToe()
        state = env.reset()
        env.render()

        actions_player_1 = []
        actions_player_2 = []

        while not game_over:
            for player_num, player_type in enumerate(players, 1):
                if player_type == 'human':
                    action = TicTacToe.human_move(env)
                else:
                    action = agent.choose_action(state, env)
                    print(f"AI Player {player_num} Move: {action}")

                if player_num == 1:
                    actions_player_1.append(action)
                else:
                    actions_player_2.append(action)

                next_state, done, winner = env.step(action, player_num)
                env.render()

                # Train the agent after each move
                reward = 0
                agent.train(state, action, reward, next_state, done)

                state = next_state

                if done:
                    game_over = True
                    if winner == 0:
                        print("It's a draw!")
                        game_outcome = (0, 0)
                    else:
                        print(f"Player {winner} wins!")
                        if winner == 1:
                            game_outcome = (1, 0)
                        else:
                            game_outcome = (0, 1)

                    plotter.add_game(actions_player_1, actions_player_1, game_outcome)

                    print("Episodes", episodes)
                    if episodes is not None:
                        episodes -= 1
                        if episodes <= 0:
                            wanna_play_game = False
                            save_model(agent.model)
                        break
                    else:
                        continue_playing = input("Do you want to play another game? (Y/N): ").strip().lower()
                        if continue_playing != 'y':
                            save_model(agent.model)
                            wanna_play_game = False
                        break
    plotter.plot_predictions()


def initialize_game():
    state_size = (3, 3, 1)
    action_size = 9
    agent = DQNAgent(state_size, action_size)
    episodes = None

    players = []
    for i in range(1, 3):
        player_type = input(f"Choose player {i} (human/ai): ").strip().lower()
        players.append(player_type)

    if 'human' not in players:
        episodes = int(input(f"How many episodes do you want AI to play? "))

    # Play the game
    play_game(agent, players, episodes)

