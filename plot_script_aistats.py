import gym
import numpy as np
import os
# from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
# from stable_baselines.sac.policies import MlpPolicy as MlpPolicySAC
# from stable_baselines.ddpg.policies import MlpPolicy as MlpPolicyDDPG
# from stable_baselines.common import make_vec_env
# from stable_baselines import PPO2
# from stable_baselines import SAC
# from stable_baselines import DDPG
# from stable_baselines import TRPO
# from stable_baselines import A2C
# from stable_baselines import ACKTR
# import gym_ct_cartpole
# from stable_baselines.common.env_checker import check_env
# from stable_baselines.common.evaluation import evaluate_policy
# from stable_baselines.bench import Monitor

import matplotlib.pyplot as plt
import pandas
import seaborn as sns


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Experiment config:
# environment rk_cartpole_swingupsafe_dm_v1
# previous model alias: best_model_bestrewardsuccess_5M

# hyperparameters for PPO2
# also, ADAM parameter beta_2 = 0

# rew_v = [700, 1300, 1400, 1500, 1600, 1800, 2001]
# lr_v = [1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4, 5e-5]
#
#
# # k = (lr_0 - lr_f) / lr_f
# def lr_rew_schedule(percent_left, rew):
#     for (r, lr) in zip(rew_v, lr_v):
#         if rew < r:
#             return lr
#     return 0.0  # This should never happen anyway...
#
#
# model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.000001,
#              noptepochs=10, learning_rate=lr_rew_schedule, nminibatches=8,
#              lam=0.995, verbose=1,
#              tensorboard_log=board_dir,
#              seed=seed)  # highest consistent reward ppo2_52 on swingup


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_name(_ode_sol, _task, _rew):
    return 'ct_{}_cartpole_{}_{}'.format(_ode_sol, _task, _rew)
ver = 'v1'
paper_dir = "./aistatsappendix"
trainingdata = "ppo2trainingdata.csv"

sns.set()
ppo2_curve = pandas.read_csv('{}/{}'.format(paper_dir, trainingdata), skiprows = 1, header = 0, index_col = False)

ppo2_curve.rename(columns ={'r': 'reward', 'l':'episode length', 't':'wallclock'}, inplace=True)
ppo2_curve['time steps'] = ppo2_curve['episode length'].cumsum()

f,ax = plt.subplots(figsize = (16,6))
sns.lineplot(x='time steps', y='reward', data=ppo2_curve , ax=ax)
plt.savefig('ppo2learningcurve.eps')
plt.show()