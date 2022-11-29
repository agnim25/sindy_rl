import gym
import numpy as np
import matplotlib.pyplot as plt
import itertools

import pysindy as ps
from pysindy import SINDy
import matplotlib.pyplot as plt
from pysindy.feature_library import *
from pysindy.differentiation import *
import numpy as np

def runEpisodeBB(heuristic=True, agent=None):
    env = gym.make("MountainCarContinuous-v0")
    obs = env.reset()
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
            action = env.action_space.sample() if np.random.uniform() < 0.2 else action
        else:
            action = agent.select_action(obs, evaluate=True)
        #temp = 10 if action == 1 else -10
        state_action.append([*obs, action])
        states.append([*obs])
        actions.append(action)
        next_obs, reward, done = env.step(action)[:3]
        obs = np.copy(next_obs)
        if done:
            obs, _ = env.reset()
            num_episodes += 1
            if num_episodes == 1:
                break
    # print("Number of episodes in data collection ", num_episodes)
    return np.array(state_action), np.array(states), np.array(actions)
    
"""
Create the transition function
"""

def create_transitionfunction(agent=None, num_trajs=10):
    print("Creating the transition function")
    states, actions = [], []
    for i in range(num_trajs):
        _, state, action = runEpisodeBB(agent)
        states.append(state)
        actions.append(action)
    lib = ps.ConcatLibrary([ps.PolynomialLibrary(), ps.FourierLibrary()])
#     optimizer = ps.STLSQ(threshold=0.001)
    optimizer = ps.SR3(threshold=0.0001, thresholder='l1',trimming_fraction=0.1,max_iter=10000)
#     der = ps.SINDyDerivative()
    der = ps.SmoothedFiniteDifference()
    model = SINDy(discrete_time=True, feature_library=lib, differentiation_method=der,
                  optimizer=optimizer)
    model.fit(states, u=actions, multiple_trajectories=True)
    return model

def test_transitionfunction(model,agent=None,num_trajs=1):
    states, actions = [], []
    for i in range(num_trajs):
        _, state, action = runEpisodeBB(agent)
        states.append(state)
        actions.append(action)
    score = model.score(states,u=actions,multiple_trajectories=True)
    print("Test data score=", score)
