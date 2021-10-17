import random

import gym
from gym.envs.classic_control import CartPoleEnv

from utils_gym import test_agent

"""
Atari envs Actions decsription
Index	Action	Description
0	NOOP	No operation, do nothing.
1	FIRE	Press the fire button without updating the joystick position
2	UP	Apply a Δ-movement upwards on the joystick
3	RIGHT	Apply a Δ-movement rightward on the joystick
4	LEFT	Apply a Δ-movement leftward on the joystick
5	DOWN	Apply a Δ-movement downward on the joystick
6	UPRIGHT	Execute UP and RIGHT
7	UPLEFT	Execute UP and LEFT
8	DOWNRIGHT	Execute DOWN and RIGHT
9	DOWNLEFT	Execute DOWN and LEFT
10	UPFIRE	Execute UP and FIRE
11	RIGHTFIRE	Execute RIGHT and FIRE
12	LEFTFIRE	Execute LEFT and FIRE
13	DOWNFIRE	Execute DOWN and FIRE
14	UPRIGHTFIRE	Execute UP and RIGHT and FIRE
15	UPLEFTFIRE	Execute UP and LEFT and FIRE
16	DOWNRIGHTFIRE	Execute DOWN and RIGHT and FIRE
17	DOWNLEFTFIRE	Execute DOWN and LEFT and FIRE"""

#
# env = gym.make('Pong-v0')
# env.action_repeat_probability = 0
#
# o = env.reset()
# x = o
#
# assert False

# env.render(mode='human')

class AtariRandomAgent:
    def act(self, observation):
        return random.randint(0,1)

env2 = CartPoleEnv()

test_agent(env2, AtariRandomAgent(), FPS=20)

