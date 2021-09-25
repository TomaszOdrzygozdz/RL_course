import random

import numpy as np

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

class CatrtPoleObservationBucketer:
    def __init__(self, n_buckets):
        self.bins = np.array([norm.ppf(x) for x in np.linspace(0,1,n_buckets)[1:-1]])

    def observation_to_bucket(self, observation):
        return np.digitize(observation, self.bins)


class WeakAgent:
    def __init__(self):
        self.bucketer = CatrtPoleObservationBucketer(25)

    def act(self, observation):
        ### Here write the code for strategy ###
        # return random.randint(0,1)
        print(f'obs = {observation} binned = {self.bucketer.observation_to_bucket(observation)}')
        return random.randint(0,1)
        # if observation[2] < 0:
        #     return 0
        # else:
        #     return 1

class CrossEntropyAgent:
    def __init__(self):
        self.experience_buffer = []
        self.bucketer = CatrtPoleObservationBucketer(25)
        # policy table keep the probability of choosing action=0
        self.policy_table = {}

    def act(self, observation):
        bucketized_obs = self.bucketer.observation_to_bucket(observation)
        if  bucketized_obs not in self.policy_table:
            return random.randint(0,1)
        else:
            if random.random() < self.policy_table[bucketized_obs]:
                return 0
            else:
                return 1

    def train(self, n_episodes, best_percent=50):
        data = test_agent(agent_class=WeakAgent, n_episodes=n_episodes,render=False)
        policy_update =
        for episode in data:
            for


data = test_agent(agent_class=WeakAgent, n_episodes=2,render=False)
print(data)