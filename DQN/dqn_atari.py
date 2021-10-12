import numpy as np
import random

import gym
from keras.models import Model
from tensorflow.keras.optimizers import Adam

from utils_gym import test_agent
from keras.layers import Input, Conv2D, Dense, BatchNormalization, GlobalAveragePooling2D, Flatten, Concatenate

random_to_action = {
    0: 2,
    1: 5}

action_to_one_hot = {2: np.array([1,0]),
                     5: np.array([0,1])}

def make_q_network(num_layers=5, kernel_size=(5,5), batch_norm=True, learning_rate=1e-3):
    dim = (210, 160, 3)
    input_observation = Input(dim)
    input_action = Input(2)
    layer = input_observation
    for _ in range(num_layers):
        layer = Conv2D(
            filters=64,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            strides=3
        )(layer)

        if batch_norm:
            layer = BatchNormalization()(layer)

    # layer = GlobalAveragePooling2D()(layer)
    obs_layer = Flatten()(layer)
    layer = Concatenate()([obs_layer, input_action])
    layer = Dense(250, activation='relu')(layer)
    layer = Dense(250, activation='relu')(layer)
    output = Dense(1)(layer)

    model = Model(inputs=[input_observation, input_action], outputs=output)
    model.compile(
                    loss='mse',
                    optimizer=Adam(learning_rate=learning_rate)
                )
    return model


class DQNAgent:
    def __init__(self):
        self.q_network = make_q_network()
        self.experience_buffer = []
        self.epsilon = 2

    def evaluate(self, observation, action):
        assert action in action_to_one_hot.keys(), 'action must be 2 (up) or (5) down'
        return self.q_network.predict([np.array([observation]), np.array([action_to_one_hot[action]])])[0][0]

    def act(self, observation):
        if random.random() < self.epsilon:
            return random_to_action[random.randint(0,1)]
        else:
            q_2 = self.evaluate(observation, 2)
            q_5 = self.evaluate(observation, 5)
            if q_2 >= q_5:
                return 2
            else:
                return 5

    def episodes_to_data(self, episodes):
        x_obs = []
        x_act = []
        for episode in episodes:
            transitions = episode.transitions_list




env = gym.make('Pong-v0')
env.action_repeat_probability = 0

obs = env.reset()

print(obs.shape)

dupa = DQNAgent()

print(dupa.evaluate(obs, 1))

# q_net = make_q_network()
# q = q_net.predict([np.array([obs]), np.array([[0,1]])])
# print(q)

test_agent(env, DQNAgent(None), FPS=20)

