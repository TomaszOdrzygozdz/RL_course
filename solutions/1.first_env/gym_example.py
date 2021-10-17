import random
from time import sleep
import gym

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


class ExampleAgent:
    def act(self, observation):
        ### Here write the code for strategy ###
        # return random.randint(0,1)
        if observation[2] < 0:
            return 0
        else:
            return 1

def test_agent_on_cartpole(agent_class, n_episodes=1, time_range=100, render=True, FPS=10):
    agent = agent_class()
    env = gym.make('CartPole-v0')
    total_reward = 0
    for i_episode in range(n_episodes):
        observation = env.reset()
        for t in range(time_range):
            if render:
                env.render(mode='human')
                sleep(1/FPS)
            action = agent.act(observation)
            observation, reward, done, info = env.step(action)
            print(f'velo = {observation[1]}, ang_velo = {observation[3]}')
            total_reward += reward
            if done:
                print('Episode finished after {} timesteps'.format(t+1))
                break
    env.close()
    print(f'Average reward per episode = {total_reward/n_episodes}')

# To visualize the agents performance:
test_agent_on_cartpole(agent_class=ExampleAgent,
           n_episodes=1,
           time_range=150,
           render=True,
           FPS=10)

