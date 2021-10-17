import numpy as np
import random

import gym
from gym.envs.classic_control import CartPoleEnv
from keras.models import Model
from tensorflow.keras.optimizers import Adam

from utils_gym import test_agent
from keras.layers import Input, Conv2D, Dense, BatchNormalization, GlobalAveragePooling2D, Flatten, Concatenate

# random_to_action = {
#     0: 2,
#     1: 5}

action_to_one_hot = {0: np.array([1,0]),
                     1: np.array([0,1])}

def make_q_network(num_layers=2, batch_norm=False, learning_rate=1e-3):
    input_observation = Input(4)
    input_action = Input(2)
    layer = input_observation
    for _ in range(num_layers):
        layer = Dense(50,activation='relu')(layer)

        if batch_norm:
            layer = BatchNormalization()(layer)

    # layer = GlobalAveragePooling2D()(layer)
    obs_layer = Flatten()(layer)
    layer = Concatenate()([obs_layer, input_action])
    layer = Dense(50, activation='relu')(layer)
    layer = Dense(50, activation='relu')(layer)
    output = Dense(1)(layer)

    model = Model(inputs=[input_observation, input_action], outputs=output)
    model.compile(
                    loss='mse',
                    optimizer=Adam(learning_rate=learning_rate)
                )
    return model

def make_q_network_convnet(num_layers=5, kernel_size=(5,5), batch_norm=True, learning_rate=1e-3):
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
    layer = Dense(100, activation='relu')(layer)
    layer = Dense(100, activation='relu')(layer)
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
        self.experience_buffer_limit = 10000
        self.alpha = 0.9
        self.epsilon = 0.1
        self.gamma = 0.995

    def evaluate(self, observation, action):
        assert action in action_to_one_hot.keys(), 'action must be 2 (up) or (5) down'
        return self.q_network.predict([np.array([observation]), np.array([action_to_one_hot[action]])])

    def evaluate_both_actions(self, observation):
        q_2, q_5 = self.q_network.predict([np.array([observation, observation]),
                                np.array([action_to_one_hot[0], action_to_one_hot[1]])]).reshape(1,2)[0]
        return q_2, q_5

    def act(self, observation):
        if random.random() < self.epsilon:
            return random.randint(0,1)
        else:
            q_0, q_1 = self.evaluate_both_actions(observation)
            if q_0 >= q_1:
                return 0
            else:
                return 1

    def add_to_buffer(self, episodes):
        self.experience_buffer.extend(list(episodes.values()))
        if len(self.experience_buffer) > self.experience_buffer_limit:
            self.experience_buffer = self.experience_buffer[-self.experience_buffer_limit:]

    def calculate_targets(self):
        observation_data = []
        action_data = []
        reward_data = []
        next_observation_data = []
        for episode in self.experience_buffer:
            for transition in episode.transitions_list:
                observation_data.append(transition.observation)
                action_data.append(action_to_one_hot[transition.action])
                reward_data.append(transition.reward)
                next_observation_data.append(transition.next_observation)

        n_samples = len(observation_data)
        observation_data = np.array(observation_data)
        action_data = np.array(action_data)
        reward_data = np.array(reward_data)
        print(f'reward_data = {reward_data}')
        old_q_data = self.q_network.predict([np.array(observation_data), action_data]).reshape(1,n_samples)[0]
        all_up = np.array([action_to_one_hot[0] for _ in range(n_samples)])
        all_down = np.array([action_to_one_hot[1] for _ in range(n_samples)])
        q_data_2 = self.q_network.predict([np.array(next_observation_data), all_up]).reshape(1,n_samples)[0]
        q_data_5 = self.q_network.predict([np.array(next_observation_data), all_down]).reshape(1,n_samples)[0]
        best_q = np.maximum(q_data_2, q_data_5)
        #Bellmans equation
        new_q_values = (1-self.alpha)*old_q_data +  self.alpha*(reward_data + self.gamma*best_q)
        return observation_data, action_data, new_q_values

    def train_one_epoch(self, num, n_episodes):
        episodes = test_agent(env, agent, n_episodes=n_episodes, render=False, print_info=False)
        total_rewards = np.array([episode.total_reward for episode in episodes.values()])
        print(f'Epoch {num} | Mean reward = {np.mean(total_rewards)}')
        self.add_to_buffer(episodes)
        observation_data, action_data, new_q_values = self.calculate_targets()
        self.q_network.fit(x=[observation_data, action_data], y=new_q_values)

    def train(self, n_epochs, episodes_per_epoch):
        for epoch_num in range(n_epochs):
            self.train_one_epoch(epoch_num, episodes_per_epoch)
        self.q_network.save('q_network')


env = CartPoleEnv()
env.action_repeat_probability = 0

# obs = env.reset()


agent = DQNAgent()
agent.train(100,1)
test_agent(env, DQNAgent(), FPS=20)
# q_net = make_q_network()
# q = q_net.predict([np.array([obs]), np.array([[0,1]])])
# print(q)
#
# episodes = test_agent(env, agent, n_episodes=1, render=False, print_info=False)
# agent.add_to_buffer(episodes)
# agent.calculate_targets()

