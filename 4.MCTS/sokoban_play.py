from gym_sokoban.envs import SokobanEnv
import matplotlib.pyplot as plt
import numpy as np
from gym_sokoban.envs.sokoban_env_fast import SokobanEnvFast

class HashableNumpyArray:
    hash_key = np.random.normal(size=1000000)

    def __init__(self, np_array):
        assert isinstance(np_array, np.ndarray), \
            'This works only for np.array'
        assert np_array.size <= self.hash_key.size, \
            f'Expected array of size lower than {self.hash_key.size} ' \
            f'consider increasing size of hash_key.'
        self.np_array = np_array
        self._hash = None

    def __hash__(self):
        if self._hash is None:
            flat_np = self.np_array.flatten()
            self._hash = int(np.dot(
                flat_np,
                self.hash_key[:len(flat_np)]) * 10e8)
        return self._hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __ne__(self, other):
        return not self.__eq__(other)

def state_to_pic(env):
    return env.render(mode='rgb_array')

def save_state_sokoban(env, file_name, title=None):
    pic = state_to_pic(env)
    plt.clf()
    if title is not None:
        plt.title(title)
    plt.imshow(pic)
    plt.savefig(file_name)

def show_state_sokoban(env):
    pic = state_to_pic(env)
    plt.clf()
    plt.imshow(pic)
    plt.show()

#
# env = SokobanEnvFast()
# env.reset()
# show_state_sokoban(env)
