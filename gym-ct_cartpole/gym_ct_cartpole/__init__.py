
import math
import numpy as np
import copy
from gym.envs.registration import register
from .dm_reward_util import sigmoids, tolerance
from .wrappers import CosineSineObservation


"""
Below are the default parameters for the configuration of the zoh-discretized continuous-time
cart-pole system. 
  
"""

"""
System Parameters:

    "mass_cart"             mass of cart in kg
    "mass_pole"             mass of pole in kg
    "xdd_max"               max acceleration threshold
    "friction_cart"         friction coefficient b_x of friction force -b*x_dot acting on the cart
    "friction_pole"         friction coefficient b_th of friction torque -b*th_dot acting in joint
    "len_pole"              length of pole in meters
    "max_force"             max force allowed in Newton
    "scale_action"          scale action of agent by this factor
    "task"                  cart-pole task: ('balance', 'swingup', 'swingupsafe'):                 
            'balance'       Cart-Pole starts upward position and is reset if position x or angle th go out of bounds
            'swingup'       Cart-Pole starts in downward position and is reset if position x goes out of bound 
                            th_max is ignored in this task
            'swingupsafe'   Same as 'swingup', but environment has a "safety"-policy included it switches to,
                            when system is close to violating the cart-position constraint.
                            While the environment executes the internal safety policy, it ignores any actions 
                            proposed by the agent.
                            
"""

sys_param = {"mass_cart": 1.0,
             "mass_pole": 0.1,
             "xdd_max": 0.5 * 9.81,
             "friction_cart": 0.0,
             "friction_pole": 0.0,
             "len_pole": 1,
             "max_force": 200,
             "scale_action": 10,
             "task": 'balance',
             }

"""
Simulation Parameters:

    "zoh_dt"                    sampling-time of zero-order hold actuation
    "ode_method"                integration method to solve continuous-time dynamics ('euler', 'rk23', 'rk45', 'rk78') :
            'euler' is just euler-step (used in OpenAI gym cartpole)
            'rk23' uses scipy.solve_ivp ( ..., method = 'RK23')
            'rk45' uses scipy.solve_ivp ( ..., method = 'RK45')
    "disturbance_type"          type of i.i.d additive disturbance: ('gaussian', 'uniform')
    "disturbance_mag"           disturbance magnitude (standard-deviation for 'gaussian' or range for 'uniform')
    "disturbance_scale"         scaling w.r.t to full state [x, th, xd, thd]
    "ode_int_steps"             how many equidistant ode integration steps to run (using rk) to compute sub-samples
"""

sim_param = {"zoh_dt": 0.02,
             "ode_method": None,
             "disturbance_type": 'gaussian',
             "disturbance_mag": 0.0,
             "disturbance_scale": [1.0, 1.0, 1.0, 1.0],
             "ode_int_steps": None}

""" 
Observation Parameters:

    "sub_samples"               number of sub-samples collected during zoh-window. 
                                Example: sub_samples = 3 and zoh_dt = 1 means that in the first zoh_window,
                                we collect measurements of [x, th, xd, thd] at times
                                                t= 0, 0.25, 0.5, 0.75, 1 
                                and stack them together in a (,20) ndarray
                                if sub_samples = 0, then the observations are just measurements of 
                                beginning and the end of the zoh-window
    "noise_type"                type of i.i.d. observational noise to sub-samples observ.
    "noise_mag"                 noise magnitude (standard-deviation for 'gaussian' or range for 'uniform')
    "noise_scale"               scaling w.r.t to full state [x, th, xd, thd]                                              
"""

obs_param = {"sub_samples": None,
             "noise_type": 'gaussian',
             "noise_mag": 0.0,
             "noise_scale": [1.0, 1.0, 1.0, 1.0]}

"""
Safety Parameters:

    "x_max"         max cart position tolerated before declared unsafe
    "theta_max"     max pole angle deviation from upright position (theta = pi) before declared unsafe
                    This option is ONLY used for balancing task 
"""

safety_param = {"x_max": 0.7,
                "theta_max": (25/180)*3.14}  # (25/180)*3.14

"""
Reset Parameters:

    "task_length"           task length in sec, then task is reset. (not necessarily system state though)
    "reset_state"           False == task get's reset but state is not, True == both task and state get reset
"""

reset_param = {"task_length": 40.0,
               "reset_state": True}

# dictionary that configures the gym environment


"""
Definition of Safe Policy

    action_out = safepolicy(state, action_in)
    
"""

xdd_max = sys_param['xdd_max']

Mm = 4.0  # max cart mass
mm = 1.0  # max pole mass
lm = 1.0  # max pole length
g = 9.81
dt_zoh = sim_param['zoh_dt']
f_max = sys_param['max_force']
x_max = safety_param['x_max']
x_buffer = x_max-0.1
use_sindy = True


def safe_barrier_signed(_x, _xd):
    return (_xd * math.fabs(_xd)) / (2.0 * xdd_max) + _x

def safe_barrier_unsigned(_x, _xd):
    return math.fabs(safe_barrier_signed(_x, _xd))

def safety_filter(z, force_in):
    """
    determines if control action can be permitted or if it needs to be overwritten and switches to back_up_policy
    :param z:
    :param force_in:
    :return:
    """
    x, th, xd, thd = z
    xdd_des = -xdd_max * np.sign(xd)
    def back_up_policy(_z):
        _x, _th, _xd, _thd = _z
        s = math.sin(_th - math.pi)
        c = math.cos(_th - math.pi)
        force_des = xdd_des * (Mm + mm * (s ** 2)) + np.sign(xdd_des) * (
                mm * lm * (thd ** 2) * np.fabs(s) + mm * g * math.fabs(s * c))
        return np.clip(force_des, -f_max, f_max)

    if (safe_barrier_unsigned(x, xd) < x_buffer) and \
            (safe_barrier_unsigned(x + xd * dt_zoh, xd) < x_buffer):
        return force_in, False  # no safety overwrite occurred
    else:
        return back_up_policy(z), True  # safety overwrite occurred


"""
Definition of Reward Function

    r = reward(state, action)

"""
#for LQR
qmat = np.diag(np.array([.3, 1, 0.05, 0.05]), 0)
rmat = np.array([0.01])

def dm_reward(state, action):
    """ Adaptation of DeepMind Control Suite Reward"""
    x, th, xd, thd = state
    upright = (math.cos(th-math.pi) + 1) / 2
    upright = tolerance(-math.cos(th), bounds=(1, 1), margin=2, value_at_margin=0.01)
    centered = tolerance(x, margin=2, value_at_margin=0.01)
    centered = (1 + centered) / 2
    small_control = tolerance(action, margin=1,
                                   value_at_margin=0.1,
                                   sigmoid='quadratic')
    small_control = (4 + small_control) / 5
    small_velocity = tolerance(thd, margin=3, value_at_margin=0.2)
    small_xvelocity = tolerance(xd, margin=0.2, value_at_margin=0.2)
    small_velocity = (1 + small_velocity) / 2
    small_xvelocity = (4 + small_xvelocity) / 5
    return upright * small_control * small_velocity * centered * small_xvelocity

def energy_barrier_reward(state, action):
    x, th, xd, thd = state
    potential = 40 * (math.cos(th) + 1.0)
    kinetic = xd ** 2 + 0.1 * thd ** 2
    ## tanh_barrier becomes positive when too close to wall
    tanh_barrier = math.tanh(2 * (safe_barrier_unsigned(x, xd) - 0.9 * x_buffer))
    if tanh_barrier > 0:
        tanh_barrier = tanh_barrier * 100
    ## kinetic barrier becomes positive when above 5
    sigmoid_kinetic = math.tanh(kinetic - 5)
    return potential - 3 * kinetic - tanh_barrier
    # return -tanh_barrier

def lqr_reward(state, action):
    (x, th, xd, thd) = state
    z = np.array([x, math.acos(math.cos(th-math.pi))/math.pi, xd, thd])
    u = np.array([action])
    lqr = z @ qmat @ z + rmat[0]*action**2

    return math.exp(-lqr)

def cos_reward(state, action):
    """ Taken from DDPG paper """
    (x, th, xd, thd) = state
    return (1-math.cos(th))/2

def one_reward(state, action):
    """ Taken from OpenAI Gym """
    return 1.0



""" Combine the Settings Dictionary """

config = {"sys": sys_param,
          "sim": sim_param,
          "obs": obs_param,
          "safety": safety_param,
          "reset": reset_param,
          "safety_filter": safety_filter,
          "reward_function": None}

config_balance = copy.deepcopy(config)
config_balance_sparse = copy.deepcopy(config)
config_balance_sparse["reward_function"] = cos_reward

config_swingup = copy.deepcopy(config)
config_swingup_sparse = copy.deepcopy(config)
config_swingup["sys"]["task"] = 'swingup'
config_swingup_sparse["sys"]["task"] = 'swingup'
config_swingup_sparse["reward_function"] = cos_reward

config_swingupsafe = copy.deepcopy(config)
config_swingupsafe["sys"]["task"] = 'swingupsafe'

ode_sol = ('rk', 'eu')
task = ('balance', 'swingup', 'swingupsafe')
rew = ('dm', 'cos', 'one')
envs = ['CartPole', 'MountainCar']
env_names = ('cartpole', 'mountaincar')

def get_name(_env, _ode_sol, _task, _rew):
    return '{}_{}_{}_{}'.format(_ode_sol, _env, _task, _rew)

## Register different settings of environment

for i in range(len(envs)):
    for s_o in ode_sol:
        for s_t in task:
            for s_r in rew:
                config_i = copy.deepcopy(config)

                if s_o == 'rk':
                    config_i['sim']['ode_method'] = 'rk4'
                    config_i['sim']['ode_int_steps'] = 4
                    config_i['obs']['sub_samples'] = 0
                elif s_o == 'eu':
                    config_i['sim']['ode_method'] = 'euler'
                    config_i['sim']['ode_int_steps'] = 1
                    config_i['obs']['sub_samples'] = 0

                config_i['sys']['task'] = s_t
                if s_r == 'dm':
                    config_i['reward_function'] = dm_reward
                elif s_r == 'cos':
                    config_i['reward_function'] = cos_reward
                elif s_r == 'one':
                    config_i['reward_function'] = one_reward

                if envs[i] == 'CartPole':
                    register(
                        id='ct_{}-v0'.format(get_name(env_names[i], s_o, s_t, s_r)),
                        entry_point='gym_ct_cartpole.envs:CTCartPoleEnv',
                        kwargs={'config': config_i, 'use_sindy': use_sindy}
                    )
                elif envs[i] == 'MountainCar':
                    register(
                        id='ct_{}-v0'.format(get_name(env_names[i], s_o, s_t, s_r)),
                        entry_point='gym_ct_cartpole.envs:CTMountainCarEnv',
                        kwargs={'use_sindy': use_sindy}
                    )

x_max_2 = 2
def dm_reward_2(state, action):
    """ Adaptation of DeepMind Control Suite Reward"""
    x, th, xd, thd = state
    upright = (math.cos(th-math.pi) + 1) / 2
    upright = tolerance(-math.cos(th), bounds=(1, 1), margin=2, value_at_margin=0.01)
    centered = tolerance(x, margin=x_max_2, value_at_margin=0.01)
    centered = (1 + centered) / 2
    small_control = tolerance(action, margin=1,
                                   value_at_margin=0.1,
                                   sigmoid='quadratic')
    small_control = (4 + small_control) / 5
    small_velocity = tolerance(thd, margin=3, value_at_margin=0.2)
    small_xvelocity = tolerance(xd, margin=0.2, value_at_margin=0.2)
    small_velocity = (1 + small_velocity) / 2
    small_xvelocity = (4 + small_xvelocity) / 5
    return upright * small_control * small_velocity * centered * small_xvelocity

for s_o in ode_sol:
    for s_t in task:
        for s_r in rew:
            config_i = copy.deepcopy(config)

            if s_o == 'rk':
                config_i['sim']['ode_method'] = 'rk4'
                config_i['sim']['ode_int_steps'] = 4
                config_i['obs']['sub_samples'] = 0
            elif s_o == 'eu':
                config_i['sim']['ode_method'] = 'euler'
                config_i['sim']['ode_int_steps'] = 1
                config_i['obs']['sub_samples'] = 0

            config_i['sys']['task'] = s_t
            config_i['safety']['x_max'] = x_max
            if s_r == 'dm':
                config_i['reward_function'] = dm_reward_2
            elif s_r == 'cos':
                config_i['reward_function'] = cos_reward
            elif s_r == 'one':
                config_i['reward_function'] = one_reward


            if envs[i] == 'CartPole':
                register(
                    id='ct_{}-v1'.format(get_name(env_names[i], s_o, s_t, s_r)),
                    entry_point='gym_ct_cartpole.envs:CTCartPoleEnv',
                    kwargs={'config': config_i, 'use_sindy': use_sindy}
                )

                config_i2 = copy.deepcopy(config_i)
                config_i2['reset']['reset_state'] = False

                register(
                    id='ct_{}-v2'.format(get_name(s_o, s_t, s_r)),
                    entry_point='gym_ct_cartpole.envs:CTCartPoleEnv',
                    kwargs={'config': config_i2, 'use_sindy': use_sindy}
                )
            
            elif envs[i] == 'MountainCar':
                register(
                    id='ct_{}-v1'.format(get_name(env_names[i], s_o, s_t, s_r)),
                    entry_point='gym_ct_cartpole.envs:CTMountainCarEnv',
                    kwargs={'use_sindy': use_sindy}
                )
                register(
                    id='ct_{}-v2'.format(get_name(env_names[i], s_o, s_t, s_r)),
                    entry_point='gym_ct_cartpole.envs:CTMountainCarEnv',
                    kwargs={'use_sindy': use_sindy}
                )
            # config_i3 = copy.deepcopy(config_i2)
            # config_i3['safety']['x_max'] = False
