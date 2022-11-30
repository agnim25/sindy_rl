import copy

from gym import spaces
from gym import ObservationWrapper
from gym_custom.envs.ct_cartpole_tasks_env import CTCartPoleEnv
import math
import numpy as np
class CosineSineObservation(ObservationWrapper):
    """ Replace theta measurement with cos(theta) and sin(theta)

    Args:
        env: The ct_cartpole environment to wrap.


    """

    def __init__(self, env):
        super(CosineSineObservation, self).__init__(env)
        # assert isinstance(env, CTCartPoleEnv), "Environment not from gym_custom"
        Nsubs = self.env.observation_space.shape[0]//4 # get number of subsamples

        # [x, cos(th), sin(th), xd, thd]
        high = np.array([np.finfo(np.float32).max,
                         1.0,
                         1.0,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max])

        high_v = np.squeeze(np.tile(high, (1, Nsubs)))
        self.observation_space = spaces.Box(-high_v, high_v, dtype=np.float32)
        return
    def observation(self, observation):
        new_observation = self._cossin_observation(observation)
        return new_observation

    def _cossin_observation(self, obs):
        o_mat = np.reshape(obs, (-1, 4))
        o_mat_T = o_mat.T
        (x_v, th_v, xd_v, thd_v) = o_mat_T
        coso_mat = np.array([x_v, np.cos(th_v), np.sin(th_v), xd_v, thd_v]).T
        coso = np.squeeze(np.reshape(coso_mat, (1, -1)))
        return coso