import random

import numpy as np
from gym.envs.classic_control import CartPoleEnv

from utils_gym import test_agent
from scipy.stats import norm

""" 
    Cartpole env instruction
        Observation:
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

        Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
        """

class CartPoleObservationBucketer:
    def __init__(self, n_buckets):
        self.bins = np.array([norm.ppf(x) for x in np.linspace(0,1,n_buckets)[1:-1]])

    def observation_to_bucket(self, observation):
        return tuple(np.digitize(observation, self.bins))


class WeakAgent:
    def __init__(self):
        self.bucketer = CartPoleObservationBucketer(20)

    def act(self, observation):
        return random.randint(0,1)


class CrossEntropyAgent:
    def __init__(self):
        self.bucketer = CartPoleObservationBucketer(25)
        # policy table keep the probability of choosing action=0
        self.policy_table = {}

    def act(self, observation):
        bucketized_obs = self.bucketer.observation_to_bucket(observation)
        if  bucketized_obs not in self.policy_table:
            return random.randint(0,1)
        else:
            if random.random() < self.policy_table[bucketized_obs][0]:
                return 0
            else:
                return 1

    def train_one_epoch(self, env, n_episodes, alpha, best_percent=90, epsilon=0.1):
        data = test_agent(env, agent=self, n_episodes=n_episodes, render=False, print_info=False)
        policy_update_raw = {}
        #select best episodes
        values_list = np.array([episode.total_reward for episode in data.values()])
        percentile = np.percentile(values_list, best_percent)

        used_episodes = 0
        for episode in data.values():
            if episode.total_reward >= percentile:
                used_episodes += 1
                for transition in episode.transitions_list:
                    observation = transition.observation
                    obs_binned = self.bucketer.observation_to_bucket(observation)
                    if obs_binned not in policy_update_raw:
                        policy_update_raw[obs_binned] = [0,0]
                    policy_update_raw[obs_binned][transition.action] += 1

        policy_update = {}
        for obs, policy in policy_update_raw.items():
            sum_over_actions = policy_update_raw[obs][0] + policy_update_raw[obs][1] + 2*epsilon
            policy_update[obs] = [
                (policy_update_raw[obs][0]+epsilon)/sum_over_actions,
                (policy_update_raw[obs][1]+epsilon)/sum_over_actions
                ]

        for key, val in policy_update.items():
            if key not in self.policy_table:
                self.policy_table[key] = val
            else:
                for i in range(2):
                    self.policy_table[key][i] = (1-alpha)*self.policy_table[key][i] + alpha*policy_update[key][i]

        return np.mean(values_list), percentile


    def train(self, episodes_per_epoch, epochs, alpha, best_percent, epsilon):
        for epoch in range(epochs):
            mean_return, percentile = self.train_one_epoch(env, episodes_per_epoch, alpha, best_percent, epsilon)
            print(f'Epoch {epoch} | mean return = {mean_return}, percentile = {percentile}')



env = CartPoleEnv()
cr_agent = CrossEntropyAgent()
cr_agent.train(episodes_per_epoch=1000, epochs=20, alpha=0.5, best_percent=90, epsilon=0.1)
data = test_agent(env, cr_agent, FPS=24)
print(f'Testing agent: reward = {data[0].total_reward}')