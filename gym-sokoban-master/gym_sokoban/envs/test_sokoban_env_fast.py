import unittest
import numpy as np

from gym_sokoban.envs.sokoban_env_fast import SokobanEnvFast


class TestSokobanFastEnvSeed(unittest.TestCase):

    def test_smoke(self):
        dim_room = (8, 8)
        env = SokobanEnvFast(dim_room=dim_room, mode='one_hot')
        self.assertEqual(env.observation_space.shape, dim_room + (7,))  # 7 types of tiles
        self.assertEqual(env.action_space.n, 4)  # up, down, left, right
        obs = env.reset()
        self.assertEqual(obs.shape, env.observation_space.shape)
        (obs, _, _, _) = env.step(env.action_space.sample())
        self.assertEqual(obs.shape, env.observation_space.shape)

    def test_seed_state(self, dim_room=(10,10), max_steps=100, num_boxes=4, mode='one_hot'):
        env = SokobanEnvFast(
            dim_room=dim_room, max_steps=max_steps, num_boxes=num_boxes, mode=mode
        )
        env.reset()
        seed = np.random.randint(0,100)
        env.seed(seed)
        env.reset()
        state = env.clone_full_state().one_hot

        for _ in range(10):
            env.seed(seed)
            env.reset()
            new_state = env.clone_full_state().one_hot
            self.assertTrue((new_state == state).all())

    def test_seed_observation(self, dim_room=(10,10), max_steps=100, num_boxes=4, mode='one_hot'):
        env = SokobanEnvFast(
            dim_room=dim_room, max_steps=max_steps, num_boxes=num_boxes, mode=mode
        )
        env.reset()
        seed = np.random.randint(0,100)
        env.seed(seed)
        ob = env.reset()

        for _ in range(10):
            env.seed(seed)
            new_ob = env.reset()
            self.assertTrue((new_ob == ob).all())



if __name__ == '__main__':
    unittest.main()
