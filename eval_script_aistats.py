import gym
import numpy as np
import os
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.sac.policies import MlpPolicy as MlpPolicySAC
from stable_baselines.ddpg.policies import MlpPolicy as MlpPolicyDDPG
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines import SAC
from stable_baselines import DDPG
from stable_baselines import TRPO
from stable_baselines import A2C
from stable_baselines import ACKTR
import gym_custom
from gym_custom import CosineSineObservation
from stable_baselines.common.env_checker import check_env
# from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.bench import Monitor

from clc.clc import clc
from evaluation_clc import evaluate_clc


def get_name(_ode_sol, _task, _rew):
    return 'ct_{}_cartpole_{}_{}'.format(_ode_sol, _task, _rew)
ver = 'v1'
paper_dir = "./ppo2models"
modelname = "ppo2model.zip"
trainingdata = "ppo2trainingdata.csv"
env_name = get_name('rk', 'swingupsafe', 'dm')
version = 'v1'



env = gym.make('{}-{}'.format(env_name, version))
env.set_sindy_use(False)
print(type(env))
envppo = CosineSineObservation(env)

# m_c, m_p, l, b_x, b_th
envppo.env.change_model(2, 0.2, 0.5, 0.5, 0.5)

env.change_model(2, 0.2, 0.5, 0.5, 0.5)

print(env.subN, env.dt_zoh, env.ode_steps)
print(env.obs)

num_episodes = 10

render_bool = False
seed_1 = 12849273

seed = seed_1

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


model_dir = os.path.join(paper_dir, modelname)


from evaluate_ppo2 import evaluate_policy

model = PPO2.load(model_dir)
print("Method: {}, System: {}".format('ppo2', env_name))
rewards, episodes_len, ep_states, ep_actions = evaluate_policy(model, envppo,
                                        n_eval_episodes=num_episodes,
                                        return_episode_rewards=True,
                                        render=render_bool)
print("{}-rewards: {},\n episode lengths: {} \n".format('ppo2', rewards, episodes_len))

# not sure what state data to extract
data = {'num_episodes': num_episodes, 'states': [[y[5:] for y in x] for x in ep_states], 'actions': ep_actions, 'all_states': ep_states}

import pickle
with open('trajectories_train.pkl', 'wb') as f:
    pickle.dump(data, f)

# clc_ctrl = clc()
# rewards_clc, episodes_len_clc, obs_log, a_log = evaluate_clc(clc_ctrl, env,
#                                                      n_eval_episodes=num_episodes,
#                                                      return_episode_rewards=True,
#                                                      render=render_bool)
# print("{}-rewards: {},\n episode lengths: {} \n".format('our method', rewards_clc, episodes_len_clc))
# env.close()
