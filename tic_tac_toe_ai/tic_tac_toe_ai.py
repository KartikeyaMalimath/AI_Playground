# Tic Tac Toe game
import os
import tensorflow as tf
from tensorflow.keras.saving import save_model

import tic_tac_toe_ai.ttt_neural_network

# Creating the board
board = [' ' for _ in range(9)]


def refresh_board():
    global board
    board = [' ' for _ in range(9)]


# Function to print the board
def print_board():
    print('---------')
    for i in range(3):
        print('|', board[i * 3], '|', board[i * 3 + 1], '|', board[i * 3 + 2], '|')
        print('---------')


# Function to check if a player has won
def check_win(player):
    # Check rows
    if player == "Bot":
        board_state_marker = "O"
    else:
        board_state_marker = "X"

    for i in range(3):
        if board[i * 3] == board[i * 3 + 1] == board[i * 3 + 2] == board_state_marker:
            return True
    # Check columns
    for i in range(3):
        if board[i] == board[i + 3] == board[i + 6] == board_state_marker:
            return True
    # Check diagonals
    if board[0] == board[4] == board[8] == board_state_marker:
        return True
    if board[2] == board[4] == board[6] == board_state_marker:
        return True
    return False


def get_winning_move(player):
    if player in ["Bot_2"]:
        marker = "O"
    else:
        marker = "X"

    # Check rows
    for i in range(3):
        if board[i * 3] == board[i * 3 + 1] == marker and board[i * 3 + 2] == ' ':
            return i * 3 + 3
        elif board[i * 3] == board[i * 3 + 2] == marker and board[i * 3 + 1] == ' ':
            return i * 3 + 2
        elif board[i * 3 + 1] == board[i * 3 + 2] == marker and board[i * 3] == ' ':
            return i * 3 + 1

    # Check columns
    for i in range(3):
        if board[i] == board[i + 3] == marker and board[i + 6] == ' ':
            return i + 7
        elif board[i] == board[i + 6] == marker and board[i + 3] == ' ':
            return i + 4
        elif board[i + 3] == board[i + 6] == marker and board[i] == ' ':
            return i + 1

    # Check main diagonal
    if board[0] == board[4] == marker and board[8] == ' ':
        return 9
    elif board[0] == board[8] == marker and board[4] == ' ':
        return 5
    elif board[4] == board[8] == marker and board[0] == ' ':
        return 1

    # Check secondary diagonal
    if board[2] == board[4] == marker and board[6] == ' ':
        return 7
    elif board[2] == board[6] == marker and board[4] == ' ':
        return 5
    elif board[4] == board[6] == marker and board[2] == ' ':
        return 3

    return None


def get_player_name():
    player_name = input("Enter your name: ")
    # player_name = "test_bot"
    if player_name:
        return player_name
    else:
        get_player_name()


# Function to play the game
def play_game():
    game_close = True
    counter = 0

    game_mode = int(input("Enter 1 for Human vs Bot or 2 for Bot vs Bot: "))

    if game_mode == 1:
        player_1 = get_player_name()
        player_2 = "Bot_2"
    else:
        player_1 = "Bot_1"
        player_2 = "Bot_2"

    model_file = 'tic_tac_toe_ai/tic_tac_toe_model.keras'

    if os.path.exists(model_file):
        print("Using Pre-Trained Model")
        model = tf.keras.models.load_model(model_file)
    else:
        model = tic_tac_toe_ai.ttt_neural_network.create_model()

    while game_close:
        current_player = player_1
        game_over = False
        game_history = []

        while not game_over:
            print_board()

            if current_player == player_2:
                player_identity = "player_2"
                move = tic_tac_toe_ai.ttt_neural_network.get_nn_move(board_state=board,
                                                                     model=model,
                                                                     player_identity=player_identity)
            else:
                player_identity = "player_1"
                if game_mode == 1:
                    move = int(input(f"Player {current_player}, enter your move (1-9): "))
                else:
                    move = tic_tac_toe_ai.ttt_neural_network.get_nn_move(board_state=board,
                                                                         model=model,
                                                                         player_identity=player_identity)
            if board[move - 1] == ' ':
                board_state_marker = "O" if current_player == player_2 else "X"
                game_history.append((board.copy(), move, 0, player_identity))
                board[move - 1] = board_state_marker

                if check_win(current_player):
                    print_board()
                    print(f"Player {current_player} wins!")
                    if current_player == player_1:
                        reward_player_1 = 1
                        reward_player_2 = -1
                    else:
                        reward_player_1 = -1
                        reward_player_2 = 1
                    game_over = True
                elif ' ' not in board:
                    print_board()
                    print("It's a tie!")
                    reward_player_1 = reward_player_2 = 0
                    game_over = True
                else:
                    winning_move = get_winning_move(player=current_player)
                    if winning_move is not None:
                        # Assign a reward for blocking the opponent's winning move
                        reward_player_1 = reward_player_2 = 0.5
                    else:
                        reward_player_1 = reward_player_2 = 0
            else:
                print("Invalid move, try again.")
                if current_player == player_1:
                    reward_player_1 = -1
                    reward_player_2 = 0
                else:
                    reward_player_1 = 0
                    reward_player_2 = -1

            # Append the reward to the game history for player 1
            game_history.append((board.copy(), move, reward_player_1, player_1))

            # Append the reward to the game history for player 2
            game_history.append((board.copy(), move, reward_player_2, player_2))

            # Update the model with reinforcement learning using the game history and rewards
            tic_tac_toe_ai.ttt_neural_network.update_model_with_rl(model=model,
                                                                   game_history=game_history)

            current_player = player_2 if current_player == player_1 else player_1

        if game_mode == 1:
            print("Do you want to play again? (Y/N)")
            if input().upper() != "Y":
                game_close = False
        else:
            counter += 1
            if counter == 100:
                game_close = False

        # Save the updated model
        save_model(model, model_file)
        print("Model saved and saved successfully!")

        # Refresh the board for the next game
        refresh_board()
