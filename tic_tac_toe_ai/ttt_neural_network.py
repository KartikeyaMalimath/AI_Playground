import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from functools import partial

from keras import Input, Model
from keras import backend as K
from keras.src.layers import Dense


# Function to get the neural network's move
def get_nn_move(board_state, model, player_identity):
    preprocessed_board = preprocess_board(board_state, player_identity=player_identity)
    nn_prediction = model.predict(preprocessed_board, verbose=0)
    nn_move = np.argmax(nn_prediction) + 1  # Add 1 to convert index to move (1-9)
    # visualize_nn_prediction(board_state, nn_prediction, player_identity)
    if np.isnan(nn_move) or board_state[nn_move - 1] != ' ':
        # If the predicted move is invalid (e.g., NaN or already taken), select a random valid move
        valid_moves = [i + 1 for i, cell in enumerate(board_state) if cell == ' ']
        return random.choice(valid_moves)
    else:
        return nn_move


def visualize_nn_prediction(board_state, nn_prediction, player_identity):
    # Visualize the board state
    # print("Current Board State:")
    # for i in range(0, 9, 3):
    #     print(board_state[i:i+3])
    # print(f"Player Identity: {player_identity}")

    # Plot the neural network's predictions
    plt.bar(range(1, 10), nn_prediction[0])
    plt.xlabel('Moves (1-9)')
    plt.ylabel('Probability')
    plt.title('Neural Network Prediction Probabilities')
    plt.show()


def create_model(rewards=None):
    """
    Create and compile a neural network model for Tic Tac Toe with reinforcement learning.

    Args:
        rewards (Tensor, optional): Tensor containing rewards obtained during gameplay. Defaults to None.

    Returns:
        model (tf.keras.Model): Compiled neural network model.
    """
    # Define input layer
    inputs = Input(shape=(11,))

    # Define the rest of the model
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(9, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    if rewards is not None:
        # Define custom loss function for reinforcement learning
        def custom_loss(y_true, y_pred):
            # Custom loss function based on policy gradient method
            # y_true: Ground truth labels (not used in this example)
            # y_pred: Predicted probabilities of actions

            # Compute log probabilities of the predicted actions
            log_probs = K.log(K.clip(y_pred, K.epsilon(), 1.0))

            # Multiply log probabilities by rewards
            weighted_log_probs = log_probs * rewards

            # Compute the mean loss across samples
            loss = -K.mean(weighted_log_probs)

            return loss

        # Compile the model with custom loss function
        model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
    else:
        # Compile the model with standard loss function
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


# Function to preprocess the board state for input to the neural network
def preprocess_board(board_state, player_identity):
    # Convert board_state to a one-hot encoded array
    # 'X' -> 1, 'O' -> -1, ' ' -> 0
    board = [1 if cell == 'X' else -1 if cell == 'O' else 0 for cell in board_state]

    # Encode player identity
    player_encoding = [1, 0] if player_identity == 'player_1' else [0, 1]

    # Concatenate player identity encoding with board state
    input_features = board + player_encoding

    return np.array(input_features).reshape(1, 11)  # 9 cells + 2 player identity features


# Function to update the model using the game history
def update_model(model, game_history, player_identity):
    X_train = []
    y_train = []
    for board_state, move in game_history:
        preprocessed_board = preprocess_board(board_state, player_identity)
        X_train.append(preprocessed_board)
        y_train.append(move - 1)  # Subtract 1 to convert move to index (0-8)
    X_train = np.vstack(X_train)
    y_train = np.array(y_train)
    model.fit(X_train, y_train, epochs=1, verbose=0)  # Train for one epoch


# Define the update_model_with_rl function to update the model with reinforcement learnin
def update_model_with_rl(model, game_history):
    X_train = []
    y_train = []
    rewards = []

    # Define preprocess_board function outside the loop or use @tf.function decorator
    for board_state, move, reward, player_identity in game_history:
        preprocessed_board = preprocess_board(board_state, player_identity)
        X_train.append(preprocessed_board)
        y_train.append(move - 1)  # Subtract 1 to convert move to index (0-8)
        rewards.append(reward)

    X_train = tf.concat(X_train, axis=0)  # Concatenate preprocessed boards along axis 0
    y_train = tf.constant(y_train, dtype=tf.int32)  # Convert y_train to TensorFlow tensor
    rewards = tf.constant(rewards, dtype=tf.float32)  # Convert rewards to TensorFlow tensor

    # Train the model using the observed rewards
    model.train_on_batch(X_train, y_train, sample_weight=rewards)
