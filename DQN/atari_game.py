import random

import gym
from time import sleep

from utils_gym import test_agent

"""
Actions decsription
0	NOOP	No operation, do nothing.
1	FIRE	Press the fire button without updating the joystick position
2	UP	Apply a Δ-movement upwards on the joystick
3	RIGHT	Apply a Δ-movement rightward on the joystick
4	LEFT	Apply a Δ-movement leftward on the joystick
5	DOWN	Apply a Δ-movement downward on the joystick"""

# env = gym.make('SpaceInvaders-v0')
env = gym.make('Pong-v0')

x = env.reset()
# env.render(mode='human')

class SpaceInvadersRandomAgent:
    def act(self, observation):
        # return 0
        return random.randint(0,5)

test_agent(env, SpaceInvadersRandomAgent(), FPS=20)

