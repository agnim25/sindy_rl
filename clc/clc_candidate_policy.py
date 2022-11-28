import control
import numpy as np
import scipy.linalg as la
import math

class candidate:
    G = 9.81
    DEG_2_RAD = 0.0174533
    RAD_2_DEG = 57.2958

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

    sim_param = {"zoh_dt": 0.02,
                 "ode_method": 'euler',
                 "disturbance_type": 'gaussian',
                 "disturbance_mag": 0.0,
                 "disturbance_scale": [1.0, 1.0, 1.0, 1.0],
                 "ode_int_steps": 1}

    obs_param = {"sub_samples": 0,
                 "noise_type": 'gaussian',
                 "noise_mag": 0.0,
                 "noise_scale": [1.0, 1.0, 1.0, 1.0]}

    safety_param = {"x_max": 0.8,
                    "theta_max": (25 / 180) * 3.14}  # (25/180)*3.14

    ctrl_param = {"Q": np.diag([100.0, 1.0, 1.0, 1.0], 0),
                  "R": 1,
                  "x_buffer": 0.1,
                  "x_buffer_in": 0.1,
                  "n_g": 0.5,
                  "n_g2": 0.5}

    reset_param = {"task_length": 30.0,
                   "reset_state": True}

    config = {"sys": sys_param,
              "sim": sim_param,
              "obs": obs_param,
              "safety": safety_param,
              "reset": reset_param,
              "ctrl": ctrl_param}

    def __init__(self, par_dict=None):
        self.subN = self.config['obs']['sub_samples'] + 2 # how many measurements collected during zoh-window
        if par_dict is None:
            self.m_c = self.config['sys']['mass_cart']
            self.m_p = self.config['sys']['mass_pole']
            self.b_x = self.config['sys']['friction_cart']
            self.b_th = self.config['sys']['friction_pole']
            self.l = self.config['sys']['len_pole']
        else:
            self.m_c = par_dict['m_c']
            self.m_p = par_dict['m_p']
            self.b_x = par_dict['b_x']
            self.b_th = par_dict['b_th']
            self.l = par_dict['l']

        self.xdd_max = self.config['sys']['xdd_max']
        self.f_max = self.config['sys']['max_force']
        self.m_cp = (self.m_p + self.m_c)

        self.polemass_length = (self.m_p * self.l)
        self.subN = self.config['obs']['sub_samples'] + 2 # how many measurements collected during zoh-window
        self.dt_zoh = self.config['sim']['zoh_dt']  # zero-order hold of actuation
        self.task = self.config['sys']['task']
        self.scale_action = self.config['sys']['scale_action']
        self.x_max = self.config['safety']['x_max']
        self.x_buf = self.config['ctrl']['x_buffer']
        self.x_buf_in = self.config['ctrl']['x_buffer_in']
        self.ng2 = self.config['ctrl']['n_g2']
        self.Q = self.config['ctrl']['Q']
        self.R = self.config['ctrl']['R']
        self.K = self._getLQRgain()
        self.packed = (self.m_c, self.m_p, self.xdd_max, self.b_x, self.b_th, self.l, self.x_max, self.x_buf,
                       self.Q, self.R, self.K, self.dt_zoh, self.x_buf_in)

    def compute_force(self, z):
        # For readability:
        (x, th, xd, thd) = z
        (M, m, xdd_max, bx, bth, l, xmax, xbuff, Q, R, K, dt, xbuffin) = self.packed

        ### shift from original code
        th = th-math.pi
        ###

        g = self.G
        s = math.sin(th)
        c = math.cos(th)

        ng2 = self.ng2
        th_lqr = 30*self.DEG_2_RAD
        c_lqr = math.cos(th_lqr)
        s_lqr = math.sin(th_lqr)
        k_swing = 2

        thdd_max = abs((ng2 * g / l) * s_lqr)

        xbd = xmax - xbuff

        xdd_desired = 0.0

        ## Compute force for partial linearization to obtain (M+m*s)xdd = f
        f_cancel = - m * g * s * c + m * l * thd ** 2 * s + m * bth * thd * c + bx * xd

        ## Compute desired acceleration for LQR
        force_lqr = float(-self.K @ np.array([x, s, xd, thd]))  # desired force according to SF-LQR controller
        xdd_lqr = (force_lqr - f_cancel)/(M + m*s**2)  # desired acceleration according to SF-LQR controller

        ## Compute desired acceleration for swing-up
        w0 = math.sqrt(m * g * l / (m * l ** 2))
        E = m * g * l * (0.5 * (thd / w0) ** 2 + c)
        E0 = -m * g * l
        Edes = m * g * l
        xdd_max2 = ng2 * g
        k_rel = xdd_max / ((abs(m * g * l * c_lqr - Edes)) / (abs(E0 - Edes)))

        xdd_swingup = -np.clip(k_swing * k_rel * abs(c) * ((E - Edes) / (abs(E0 - Edes))) * np.sign(thd * c),
                               -xdd_max, xdd_max)

        cut_off = m * g * l * c_lqr

        ## Adjust acceleration of swing up if energy is close to target energy
        if E > cut_off: # If energy is close to desired, focus on getting back to center
            if abs(x) > xbd / 3:
                if xd >= 0:
                    xdd_swingup = min([xdd_swingup, -xdd_max / 2 * np.sign(x)])
                else:
                    xdd_swingup = max([xdd_swingup, -xdd_max / 2 * np.sign(x)])


        ## Check if we are close to up-right position
        ready_2_regulate = False
        if abs(thd ** 2 / thdd_max) < 2 * th_lqr:
            if math.cos(thd ** 2 / thdd_max + th * np.sign(thd)) > c_lqr:
                ready_2_regulate = True

        ## Determine whether to set acceleration for swing-up or regulation
        if ready_2_regulate and abs(xdd_lqr) < xdd_max2:
            xdd_desired = xdd_lqr
        else:
            xdd_desired = xdd_swingup

        ## Adjust desired acceleration if it is unsafe
        xbd_in = (xmax-xbuff-xbuffin)
        def insafe_inner(xdd_des):
            Vnxt = abs((1/(2*xdd_max))*(xd+xdd_des*dt)*abs(xd + xdd_des*dt) + x + xd*dt + 0.5*xdd_des*dt**2)
            Vnow = abs((1/(2*xdd_max))*xd*abs(xd) + x)
            if Vnow < xbd_in and Vnxt < xbd_in:
                return True
            else:
                return False

        if not insafe_inner(xdd_desired):
            Vnow = self.safe_barrier_signed(x, xd)
            if Vnow >= 0:
                xdd_desired = min([xdd_desired, -xdd_max * np.sign(xd)])
            else:
                xdd_desired = max([xdd_desired, -xdd_max * np.sign(xd)])

        f = (M + m * s ** 2) * xdd_desired + f_cancel
        f = np.clip(f, -self.f_max, self.f_max)

        return f

    def compute_action(self, obs):
        z = obs[-4:]  # get last 4 elements which represent (x, th, xd, thd)
        f = self.compute_force(z)
        action = f/self.scale_action
        return np.array([action]) # environments expect inputs as numpy arrays

    def safe_barrier_signed(self, _x, _xd):
        return (_xd * abs(_xd)) / (2.0 * self.xdd_max) + _x

    def safe_barrier_unsigned(self, _x, _xd):
        return abs(self.safe_barrier_signed(_x, _xd))

    def _getLQRgain(self):
        M = self.m_c
        m = self.m_p
        l = self.l
        bx = self.b_x
        bth = self.b_th
        g = self.G
        Q = self.Q
        R = self.R

        A1 = np.hstack((np.zeros((2,2)), np.eye(2)))
        Meq = [[M+m, -m*l], [-1.0, l]]
        S = [[1], [0]]
        invM = la.pinv(Meq)
        B = np.vstack((np.zeros((2, 1)), invM @ S))
        A2 = np.hstack((invM @ np.array([[0, 0], [0, g]]), invM @ np.array([[-bx, 0], [0, -bth]])))
        A = np.vstack((A1, A2))
        K, _, _ = control.lqr(A, B, Q, R)
        K = np.squeeze(K)
        return K



