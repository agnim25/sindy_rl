import numpy as np

import gym
from gym import utils
from gym.envs.mujoco import MuJocoPyEnv
from gym.spaces import Box
import pysindy as ps

class SwimmerEnv(MuJocoPyEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(self, **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)
        MuJocoPyEnv.__init__(
            self, "swimmer.xml", 4, observation_space=observation_space, **kwargs
        )
        utils.EzPickle.__init__(self, **kwargs)
        self.ob = self._get_obs()
        self.sindy_model = None
        self.steps = 0
        self.max_steps = 1000

    def step(self, a):
        self.steps += 1
        ctrl_cost_coeff = 0.0001
        if self.sindy_model is None:
            xposbefore = self.sim.data.qpos[0]
            self.do_simulation(a, self.frame_skip)
            xposafter = self.sim.data.qpos[0]
            ob = self._get_obs()
        else:
            xposbefore = self.ob[0]
            action_vec = np.array([a])
            # print(state_vec, action_vec)
            print(np.array([self.ob]).shape)
            print(np.array([self.ob]))
            print(action_vec.shape)
            ob = self.sindy_model.predict(np.array([self.ob]), u=action_vec)
            xposafter = ob[0][0]
            self.ob = ob[0]

        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = -ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl

        if self.render_mode == "human":
            self.render()
        
        done = False
        if self.steps > self.max_steps:
            done = True

        return (
            ob,
            reward,
            done,
            dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl),
        )

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat, qvel.flat])

    def reset(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv),
        )
        self.steps = 0
        return self._get_obs()

    def runEpisodeBB(self, heuristic=True, agent=None):
        env = gym.make("Swimmer-v3", exclude_current_positions_from_observation=False)
        obs = list(env.reset()[0])
        state_action = []
        states = []
        actions = []
        num_episodes = 0
        k = 2
        action = env.action_space.sample()
        action = env.action_space.sample()
        for i in range(1000):
            if heuristic:
                action = 1
                # action = env.action_space.sample() if np.random.uniform() < 0.2 else action
            if agent == None:
                curr_action = action
                action = -1*action
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs, evaluate=True)
            #temp = 10 if action == 1 else -10
            state_action.append([*obs, action])
            states.append([*obs])
            actions.append(action)
            next_obs, reward, done, _, _ = env.step(action)
            obs = np.copy(next_obs)
            if done:
                obs, _ = env.reset()
                num_episodes += 1
                if num_episodes == 1:
                    break
        # print("Number of episodes in data collection ", num_episodes)
        return np.array(state_action), np.array(states), np.array(actions)

    def create_transitionfunction(self, agent=None, num_trajs=10):
        states = []
        actions = []
        print("Creating the transition function")
        for i in range(num_trajs):
            xva, state, action = self.runEpisodeBB(agent)
            states.append(state)
            actions.append(action)
            print(len(state))

        lib = ps.ConcatLibrary([ps.PolynomialLibrary(), ps.FourierLibrary()])
        optimizer = ps.SR3(threshold=0.0001, thresholder='l1',trimming_fraction=0.1,max_iter=10000)
        #lib = PolynomialLibrary()
        der = ps.SINDyDerivative()
        der = ps.SmoothedFiniteDifference()
        model = ps.SINDy(discrete_time=True, feature_library=lib, differentiation_method=der,
                    optimizer=optimizer)
        model.fit(states, u=actions, multiple_trajectories=True)
        model.print()
        print(model.score(states, u=actions, multiple_trajectories=True))
        return model

    def use_sindy_model(self):
        self.sindy_model = self.create_transitionfunction(num_trajs=10)
        
    
