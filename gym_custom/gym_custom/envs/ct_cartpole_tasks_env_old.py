import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from scipy.integrate import solve_ivp
import pyglet


class CTCartPoleEnv(gym.Env):
    """ by Dimitar Ho, dho@caltech.edu,  28th April 2020:

    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a track.
        The system is simulated with zero-order hold actuation and N sub-samples during zoh window.
        Three tasks are available depending on configuration: (balance, swing-up, safe-swing-up)
    Observation:
        Type: Box(,4*(N + 2))
        Num	Observation
        x	Cart Position
        th	Pole Angle
        xd	Cart Velocity
        thd	Pole Velocity

    Actions:
        Type: Box(1) -  Force on the cart
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    METHODS = {'euler': 'euler', 'rk4': 'rk4', 'rk23': 'RK23', 'rk45': 'RK45', 'rk78': 'DOP853'}
    G = 9.81  # gravity constant

    def __init__(self, config):
        self.m_c = config['sys']['mass_cart']
        self.m_p = config['sys']['mass_pole']
        self.xdd_max = config['sys']['xdd_max']
        self.m_cp = (self.m_p + self.m_c)
        self.b_x = config['sys']['friction_cart']
        self.b_th = config['sys']['friction_pole']
        self.l = config['sys']['len_pole']
        self.polemass_length = (self.m_p * self.l)
        self.subN = config['obs']['sub_samples'] + 2  # how many measurements collected during zoh-window
        self.dt_zoh = config['sim']['zoh_dt']  # zero-order hold of actuation
        self.ode_steps = int(config['sim']['ode_int_steps'])
        assert config['sim']['ode_method'] in self.METHODS.keys(), "ode_method not recognized"

        self.ode_int = self.METHODS[config['sim']['ode_method']]
        self.task = config['sys']['task']
        self.action_scale = config['sys']['scale_action']

        self.seed()
        assert self.task in ('swingup', 'balance', 'swingupsafe'), "{} task not specified".format(
            self.task)
        self.reset_state = config['reset']['reset_state']
        assert isinstance(self.reset_state, bool), "{} is not boolean".format(self.reset_state)

        self.task_T = config['reset']['task_length']  # in seconds
        self.steps = 0
        self.max_steps = int(self.task_T / self.dt_zoh)
        self.state = self.np_random.uniform(low=-0.01, high=0.01, size=(4,))
        if self.task == 'balance':
            # change to start in upward position
            self.state[1] += math.pi

        # self.obs saves the current observation of the environment
        #   and in the beginning we just repeat the state subN times
        state_mat = np.tile(self.state, (self.subN, 1))
        state_vec = np.reshape(state_mat, (1, -1))
        self.obs = state_vec
        # constraints which if violated fail the episode
        self.x_max = config['safety']['x_max']
        self.safety_filter = config['safety_filter']
        self.reward_function = config['reward_function']
        if self.task == 'balance':
            # theta has a constraint only in the balance task
            self.th_max = config['safety']['theta_max']
        else:
            self.th_max = None
        self.system_outofbounds = False  # keep internal state if we are out of bounds
        self.unsafe_count = 0
        # initialize state and observation and reset to initial state
        self._reset_all()
        self.unsafe_penalty = 0  # reward penalty for crashing into walls
        self.safe_policy_penalty = 0  # reward penalty for triggering safety controller
        # LQR reward

        # State dimension is 4, so N*4 scalar sub-sample values
        high = np.ones(4 * self.subN) * np.finfo(np.float32).max

        self.f_max = config['sys']['max_force']
        self.action_max = self.f_max / self.action_scale

        self.action_space = spaces.Box(-self.action_max, self.action_max, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.viewer = None

        self.steps_beyond_done = None

    def change_model(self, new_mc, new_mp, new_lp, new_bx, new_bth):
        # Change model parameters from outside
        self.m_c, self.m_p, self.l, self.b_x, self.b_th = new_mc, new_mp, new_lp, new_bx, new_bth
        self.m_cp = self.m_c + self.m_p
        self.polemass_length = (self.m_p * self.l)
        logger.warn("Environment changed model parameters: m_c = {}, m_p = {}, "
                    "l_p = {}, b_x = {}, b_th = {} ".format(new_mc, new_mp, new_lp, new_bx, new_bth))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        action = np.clip(action, -self.action_max, self.action_max)
        state = self.state
        force_agent = self.action_scale * action[0]
        force = None
        if self.task == 'swingupsafe':  # ('swingup', 'balance', 'swingupsafe')
            force, overwrite_event = self.safety_filter(state, force_agent)
        else:
            force = force_agent

        f_zoh = np.clip(force, -self.f_max, self.f_max)
        # do a zero order hold step simulation of continuous dynamical system
        tspan = [0.0, self.dt_zoh]
        tev = np.linspace(0, self.dt_zoh, self.subN)
        zinit = state

        # safety_override = False
        ## Safety Policy
        # if too close to walls, safe policy takes over

        # %% Solve differential equation

        f_cartpole_closedloop = lambda t, z: self.fun_cartpole(t, z, f_zoh)
        obs, znext = None, None
        if self.ode_int in ('euler', 'rk4'):
            (obs, znext) = self.rk_step(f_cartpole_closedloop, zinit)
        else:
            sol = solve_ivp(f_cartpole_closedloop, tspan, zinit, t_eval=tev, method=self.ode_int)
            znext = sol.y[:, -1]
            ymat = sol.y
            obs = np.squeeze(np.reshape(ymat.T, (1, -1)))

        self.obs = obs  # safe observation internally
        self.state = znext  # safe next state
        self.steps += 1  # update step counter

        x_n, th_n, xd_n, thd_n = znext

        x_outofbounds = False
        if self.x_max is not None:
            x_outofbounds = bool(x_n < -self.x_max or x_n > self.x_max)

        th_outofbounds = False
        if self.th_max is not None:
            th_outofbounds = bool(th_n > (math.pi + self.th_max) or th_n < (math.pi - self.th_max))

        if self.task == 'balance':
            done = x_outofbounds or th_outofbounds
            self.system_outofbounds = self.system_outofbounds or x_outofbounds or th_outofbounds
        else:
            done = x_outofbounds
            self.system_outofbounds = self.system_outofbounds or x_outofbounds

        done = bool(done)

        reward = 0.0
        if not done:
            reward = self.reward_function(state, action[0])
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            logger.warn("Environment violated safety constraints. ")
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True."
                    " You should always call 'reset()' once"
                    " you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1

        if self.steps > self.max_steps:
            done = True
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0

        # as observation return all subsamples of zoh time step
        return obs, reward, done, {}

    """
    Equations of motion taken from 
    
    Russ Tedrake. Underactuated Robotics: Algorithms for Walking, Running, Swimming, Flying, and Manipulation
     (Course Notes for MIT 6.832). Downloaded from http://underactuated.mit.edu
     
    Not that th = 0 represents the downward position and th = pi represents the upward position 

    """

    def fun_cartpole(self, t, z, f_zoh):

        x, th, xd, thd = z
        c = math.cos(th)
        s = math.sin(th)

        m_c, m_p, l, bx, bth, g = self.m_c, self.m_p, self.l, self.b_x, self.b_th, self.G
        xdd_denom = m_c + m_p * s ** 2
        thdd_denom = l * (m_c + m_p * s ** 2)
        xdd_nom = f_zoh + m_p * s * (l * thd ** 2 + g * c)
        thdd_nom = -f_zoh * c - m_p * l * thd ** 2 * c * s - (m_c + m_p) * g * s
        xdd = xdd_nom / xdd_denom
        thdd = thdd_nom / thdd_denom

        dzdt = [xd, thd, xdd, thdd]
        return dzdt

    def rk_step(self, f_closed_loop, z0):
        N = self.subN
        dt = self.dt_zoh / (N - 1)  # step for subsamples
        n = self.ode_steps
        obs = z0
        z = z0
        assert self.ode_int in ('euler', 'rk4'), "ode_int method not defined in rk_step"
        for i in range(N - 1):
            z_next = None
            if self.ode_int == 'euler':
                (_, _, z_n) = self.euler(f_closed_loop, 0, z, dt, n)
                z_next = z_n
            elif self.ode_int == 'rk4':
                (_, _, z_n) = self.rk4(f_closed_loop, 0, z, dt, n)
                z_next = z_n
            obs = np.hstack((obs, z_next))
            z = z_next

        obs = np.squeeze(obs)
        return obs, z

    def rk4(self, f, t0, x0, t1, n):
        # n - rk4 steps to to compute x(t1)
        vt = np.array([0] * (n + 1))
        vt = [0] * (n + 1)
        h = (t1 - t0) / float(n)
        vt[0] = t = t0
        x = np.array(x0)
        vx = x
        for i in range(1, n + 1):
            k1 = h * np.array(f(t, x))
            k2 = h * np.array(f(t + 0.5 * h, x + 0.5 * k1))
            k3 = h * np.array(f(t + 0.5 * h, x + 0.5 * k2))
            k4 = h * np.array(f(t + h, x + k3))
            vt[i] = t = t0 + i * h
            x = x + (k1 + k2 + k2 + k3 + k3 + k4) / 6
            vx = np.vstack((vx, x))
        x_last = x
        return vt, vx, x_last

    def euler(self, f, t0, x0, t1, n):
        # n - euler steps to to compute x(t1)
        vt = np.array([0] * (n + 1))
        vt = [0] * (n + 1)
        h = (t1 - t0) / float(n)
        vt[0] = t = t0
        x = np.array(x0)
        vx = x
        for i in range(1, n + 1):
            k1 = h * np.array(f(t, x))
            vt[i] = t = t0 + i * h
            x = x + k1
            vx = np.vstack((vx, x))
        x_last = x
        return vt, vx, x_last

    def reset(self):
        self.steps_beyond_done = None
        self.steps = 0
        if self.reset_state or self.system_outofbounds:
            if self.system_outofbounds:
                self.unsafe_count += 1
            return self._reset_all()
        else:
            return self.obs

    def _reset_all(self):
        self.state = self.np_random.uniform(low=-0.01, high=0.01, size=(4,))
        self.system_outofbounds = False
        if self.task == 'balance':
            # change to start in upward position
            self.state[1] += math.pi
        # self.obs saves the current observation of the environment
        #   and in the beginning we just repeat the state subN times
        state_mat = np.tile(self.state, (self.subN, 1))
        state_vec = np.squeeze(np.reshape(state_mat, (1, -1)))
        self.obs = state_vec
        return state_vec

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400
        x_max = 2
        if self.x_max is not None:
            x_max = self.x_max
        world_width = (x_max + self.l) * 2
        scale = screen_width / world_width
        carty = screen_height / 2  # TOP OF CART
        polelen = scale * (self.l)
        polewidth = polelen / 15
        cartwidth = scale * 0.2
        cartheight = scale * 0.1
        wallwidth = 3.0
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            wallpos = x_max
            l, r, t, b = screen_width / 2.0 - scale * wallpos - wallwidth - cartwidth / 2, screen_width / 2.0 - scale * wallpos - cartwidth / 2, 0, screen_height
            self.wall_left = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.wall_left.set_color(1, 0, 0)

            l, r, t, b = screen_width / 2.0 + scale * wallpos + cartwidth / 2, screen_width / 2.0 + scale * wallpos + wallwidth + cartwidth / 2, 0, screen_height
            self.wall_right = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.wall_right.set_color(1, 0, 0)

            self.viewer.add_geom(self.wall_left)
            self.viewer.add_geom(self.wall_right)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.score_label = pyglet.text.Label("{}, {} sub-steps".format(str(self.ode_int), self.ode_steps),
                                                 font_size=36,
                                                 x=0, y=0, anchor_x='left', anchor_y='center',
                                                 color=(0, 255, 0, 255))
            self.viewer.add_geom(self.track)

            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polewidth / 2, -polelen + polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)

            self._pole_geom = pole

        if self.state is None: return None

        x, th, xd, thd = self.state
        cartx = x * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(th)

        obj = self.viewer.render(return_rgb_array=mode == 'rgb_array')
        self.score_label.draw()
        return obj

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
