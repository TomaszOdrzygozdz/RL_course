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
        self.policy_table = {}

    def act(self, observation):
        bucketized_obs = self.bucketer.observation_to_bucket(observation)
        # < YOUR CODE HERE >

    def train_one_epoch(self, env, n_episodes, alpha, best_percent=90, epsilon=0.1):
        data = test_agent(env, agent=self, n_episodes=n_episodes, render=False, print_info=False)

        # WRITE A FUNCTION THAT TRAINS ON ONE PART OF DATA AND RETURNS mean value and percentile value


    def train(self, episodes_per_epoch, epochs, alpha, best_percent, epsilon):
        for epoch in range(epochs):
            mean_return, percentile = self.train_one_epoch(env, episodes_per_epoch, alpha, best_percent, epsilon)
            print(f'Epoch {epoch} | mean return = {mean_return}, percentile = {percentile}')



env = CartPoleEnv()
cr_agent = CrossEntropyAgent()
# cr_agent.train(episodes_per_epoch=1000, epochs=20, alpha=0.5, best_percent=90, epsilon=0.1)
data = test_agent(env, cr_agent, FPS=24)
print(f'Testing agent: reward = {data[0].total_reward}')