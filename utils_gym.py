from time import sleep

import collections

Transition = collections.namedtuple('transition',
                                    ['observation',
                                     'action',
                                     'reward',
                                     'next_observation'])


Episode = collections.namedtuple('episode',
                                 ['transitions_list',
                                  'total_reward'])

def test_agent(env, agent, n_episodes=1, time_range=200, render=True, FPS=10, print_info=True):
    episodes = {}
    episode_num = 0
    for i_episode in range(n_episodes):
        total_reward = 0
        observation = env.reset()
        transitions_list = []
        for t in range(time_range):
            if render:
                env.render(mode='human')
                sleep(1/FPS)
            action = agent.act(observation)
            next_observation, reward, done, info = env.step(action)
            print(f'reward = {reward}')
            transitions_list.append(Transition(observation, action, reward, next_observation))
            observation = next_observation
            total_reward += reward
            if done:
                if print_info:
                    print('Episode finished after {} timesteps'.format(t+1))
                break
        episodes[episode_num] = Episode(transitions_list, total_reward)
        episode_num += 1
    env.close()
    if print_info:
        print(f'Average reward per episode = {total_reward/n_episodes}')
    return episodes