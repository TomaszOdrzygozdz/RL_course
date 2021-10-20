import itertools
import random
from time import sleep

import numpy as np
import tensorflow

from gym_sokoban.envs.sokoban_env_fast import SokobanEnvFast
from utils_gym import test_agent
from sokoban_play import show_state_sokoban, save_state_sokoban, HashableNumpyArray

def get_field_name_from_index(x):
    objects = {0: 'wall', 1: 'empty', 2: 'goal', 3: 'box_on_goal', 4: 'box', 5: 'agent', 6: 'agent_on_goal'}
    return objects[x]

class Value:
    def __init__(self, model_id='/home/tomek/Research/RL_course/4.MCTS/sokoban_value/12_12_3'):
        self.model = tensorflow.keras.models.load_model(model_id)

    def evaluate(self, obs):
        return self.model.predict(np.array([obs]))[0][0] + self.reward_for_boxes_on_goals(obs)

    def reward_for_boxes_on_goals(self, state):
        reward = 0
        for xy in itertools.product(list(range(state.shape[0])),
                                    list(range(state.shape[1]))):
            x, y = xy
            if get_field_name_from_index(np.argmax(state[x][y])) == 'box_on_goal':
                reward += 1
        return reward

class GreedyAgent:
    def __init__(self):
        self.value_network = Value()
        self.env = SokobanEnvFast(dim_room=(8,8), num_boxes=2)
        self.epsilon = 0.25
        self.action_num = 0
        self.seen_states = set()

    def act(self, obs):
        save_state_sokoban(self.env, f'trajectory/{self.action_num}.png')
        self.seen_states.add(HashableNumpyArray(obs))
        self.action_num += 1
        if random.random() < self.epsilon:
            return random.randint(0,3)
        else:
            self.env.restore_full_state_from_np_array_version(obs)

            full_state = self.env.clone_full_state()
            values = {i: 0 for i in range(4)}
            best_action = random.randint(0,3)
            best_value = -float('inf')
            for a in range(4):
                self.env.restore_full_state(full_state)
                observation, _, done, info = self.env.step(a)
                values[a] = self.value_network.evaluate(observation)
                if not HashableNumpyArray(observation) in self.seen_states:
                    print('unseen')
                    if values[a] > best_value:
                        best_action = a
                        best_value = values[a]
                else:
                    print('seen state')

            print(f'vals = {values} best_action = {best_action}')
            return best_action

env = SokobanEnvFast(dim_room=(12,12), num_boxes=3)
test_agent(env, GreedyAgent(), render=False, time_range=150, print_info=True)

# v = Value()
# env = SokobanEnvFast(dim_room=(12,12), num_boxes=4)
# o = env.reset()
# print(v.evaluate(o))