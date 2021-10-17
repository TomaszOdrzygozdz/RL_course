import random

import gym
import numpy as np
from gym.envs.classic_control import CartPoleEnv
from scipy.stats import norm
import matplotlib.pyplot as plt

from utils_gym import test_agent


class CartPoleObservationBucketer:
    def __init__(self, n_buckets):
        self.bins = np.array([norm.ppf(x) for x in np.linspace(0,1,n_buckets)[1:-1]])

    def observation_to_bucket(self, observation):
        # return observation
        return tuple(np.digitize(observation, self.bins))

class QLearningAgent:
    def __init__(self, epsilon=0.25, alpha=0.5, gamma=1.):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}
        self.bucketer = CartPoleObservationBucketer(30)
        self.experience_buffer = []
        self.counter = [0,0]
        self.best_q = {}

    def act(self, observation):
        bucketized_obs = self.bucketer.observation_to_bucket(observation)
        if bucketized_obs not in self.q_table:
            # print('not in')
            self.q_table[bucketized_obs] = [0,0]

        if random.random() < self.epsilon:
            return random.randint(0,1)
        else:
            if self.q_table[bucketized_obs][0] > self.q_table[bucketized_obs][1]:
                self.counter[0] += 1
                return 0
            elif self.q_table[bucketized_obs][0] < self.q_table[bucketized_obs][1]:
                self.counter[1] += 1
                return 1
            else:
                return random.randint(0, 1)

    def train_one_epoch(self, env, n_episodes):
        self.epsilon = self.epsilon*0.9999
        data = test_agent(env, agent=self, n_episodes=n_episodes, render=False, print_info=False)
        self.experience_buffer.extend(list(data.values()))
        if len(self.experience_buffer) > 1:
            self.experience_buffer = self.experience_buffer[-1:]
        policy_update_raw = {}
        # select best episodes
        values_list = np.array([episode.total_reward for episode in data.values()])
        # print(f'Mean reward = {np.mean(values_list)}')
        q_new = {}
        for episode in self.experience_buffer:
            i = 1
            last_obs = None
            for transition in episode.transitions_list:
                raw_observation = transition.observation
                observation = self.bucketer.observation_to_bucket(raw_observation)
                next_observation_raw = transition.next_observation
                next_observation = self.bucketer.observation_to_bucket(next_observation_raw)
                action = transition.action
                reward = transition.reward
                # if i == len(episode.transitions_list):
                #     reward = i
                q_new[observation] = [0,0]
                if next_observation not in self.q_table:
                    self.q_table[next_observation] = [0,0]
                q_new[observation][action] = reward + self.gamma*max([self.q_table[next_observation][0], self.q_table[next_observation][1]])
                if q_new[observation][0] > 200 or q_new[observation][1] > 200:
                    x =3
                if self.q_table[observation][0] > 200 or self.q_table[observation][1] > 200:
                    x =3

                # x = [self.q_table[next_observation][0], self.q_table[next_observation][1]]
                self.q_table[observation][action] = (1 - self.alpha) * self.q_table[observation][action] + self.alpha * q_new[observation][action]

        # for obs in q_new:
        #     for a in range(0,2):
        #         pass
        # print(self.q_table)
        # print(f'len_self_q = {len(self.q_table)} max[0] = {max([x[0] for x in self.q_table.values()])} max[1] = {max([x[1] for x in self.q_table.values()])} ')
        return np.mean(values_list)

    def train(self, env, episodes_per_epoch, epochs):
        y = []
        for epoch in range(epochs):
            mean_return = self.train_one_epoch(env, episodes_per_epoch)
            best_result = 0
            if epoch % 100 == 0:
                data = test_agent(env, agent=self, n_episodes=100, render=False, print_info=False)
                values_list = np.array([episode.total_reward for episode in data.values()])
                mean_val = np.mean(values_list)
                print(f'Epoch {epoch} | epsilon = {self.epsilon}  | mean return = {mean_val} | counter = {self.counter}')
                # print(self.q_table)
                y.append(mean_return)

        plt.plot(y)
        plt.show()

env = CartPoleEnv()
# env = gym.make("Taxi-v3")
q_agent = QLearningAgent()
q_agent.train(env, episodes_per_epoch=1, epochs=10000)
# data = test_agent(env, q_agent, FPS=5)
# print(f'Testing agent: reward = {data[0].total_reward}')
