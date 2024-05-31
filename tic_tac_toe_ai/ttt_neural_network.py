import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DQNAgent:
    def __init__(self, state_size, action_size, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, gamma=0.95, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            layers.Input(shape=self.state_size),
            layers.Conv2D(64, (2, 2), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def choose_action(self, state, env):
        valid_actions = env.available_actions()
        if np.random.rand() <= self.epsilon:
            return_move = random.choice(valid_actions)
            return return_move
        q_values = self.model.predict(state)
        valid_q_values = [q_values[0][(action[0] * 3) + action[1]] for action in valid_actions]
        return_move = valid_actions[np.argmax(valid_q_values)]
        return return_move

    @staticmethod
    def calculate_reward(state, action, step_func):
        # Retrieve the current player and opponent from the state
        current_player = 1 if np.sum(state) % 2 == 0 else 2
        opponent = 2 if current_player == 1 else 1

        # Check if the action leads to a win for the current player
        _, done, winner = step_func(action, current_player)
        if done and winner == current_player:
            return 1  # Reward for winning

        # Check if the action leads to a win for the opponent
        _, done, winner = step_func(action, opponent)
        if done and winner == opponent:
            return -1  # Reward for opponent winning

        # Check if the action blocks the opponent from winning
        _, done, winner = step_func(action, opponent)
        if done and winner == opponent:
            return 0.5  # Reward for blocking opponent's winning move

        # Otherwise, return a small negative reward to encourage exploration
        return -0.01

    def train(self, state, action, reward, next_state, done):
        state = np.reshape(state, (1, 3, 3, 1))  # Ensure proper shape
        next_state = np.reshape(next_state, (1, 3, 3, 1))  # Ensure proper shape

        target = reward
        if not done:
            target = reward + self.gamma * np.max(self.model.predict(next_state)[0])

        target_f = self.model.predict(state)
        target_f[0][action-1] = target

        self.model.fit(state, target_f, epochs=1, verbose=0)


def create_model():
    return DQNAgent(state_size=(3, 3, 1), action_size=9)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def add(self, experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)
