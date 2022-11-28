from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import polytope as pc
from cvxopt import matrix, solvers
from scipy.optimize import linprog as lp
import math


class ConsistentPolytope:
    """
    ConsistentPolytope class with following fields
        'poly'      Polytope describing current parameter constraints
        'dt'        sub-sampled time
    """
    G = 9.81
    SEED = 4082
    EPS = 0.000 # parameter precision
    def __init__(self, pmin, pmax, dt_subs, n_subs):
        np.vstack((pmin, pmax)).T
        self.poly = pc.box2poly(np.vstack((pmin, pmax)).T)  # initial box polytope
        # (A, b) = self.poly.A, self.poly.b
        # (A, b) = self.add_slack(A, b)
        # self.poly = pc.Polytope(A, b)
        self.dt = dt_subs
        self.N = n_subs
        self.n = 4
        self.p_dim = self.poly.dim
        self.np_rng = np.random.default_rng()
        self.xdd_max = 0.5*self.G
        ## Add non-negative inertia constraint

    # def addConstraints(self, x0batch, u0batch, x1batch):
    #     nD = np.shape(x0batch)[0] # how many rows
    #
    #     A1 = np.hstack(x0batch, u0batch, -np.ones((1, nD)))
    #     A2 = np.hstack(-x0batch, -u0batch, -np.ones((1, nD)))
    #     A12 = np.vstack((A1, A2))
    #
    #     b12 = np.vstack((x1batch, -x1batch))
    #
    #     newpoly = pc.Polytope(A12, b12)
    #
    #     self.poly = self.poly.intersect(newpoly) # intersect with previous constraints and reduce

    def build_features(self, obs, f):
        """
        building features for parameter vector
            p = [m_c+m_p, m_p*l, b_x, l, b_th, eta_x, eta_th] or p = [p_1, p_2] where

                p_1 = [m_c+m_p, m_p*l, b_x, l, b_th]    p_2 = [eta_x, eta_th]

                A_lp*p <= b_lp           or         [-A, -I][p_1] <= [-b]
                                                    [ A, -I][p_2] <= [ b]

        expecting obs = (,n*n_subs)

        return (A_lp, b_lp)
        """
        o_mat = np.reshape(obs, (-1, self.n))

        ## build derivatives
        od = (o_mat[-1]-o_mat[0])/((self.N-1)*self.dt)
        o = np.mean(o_mat, 0)

        (x, th, xd, thd) = o
        (_, _, xdd, thdd) = od

        #### SHIFT FROM ORIGINAL CODE
        th = th - math.pi
        ####

        s = math.sin(th)
        c = math.cos(th)

        a_xdd = np.array([xdd, thd**2*s-thdd*c, xd, 0, 0])
        a_thdd = np.array([0, 0, 0, thdd, thd])

        b_xdd = np.ndarray(shape=(1,))
        b_thdd = np.ndarray(shape=(1,))
        b_xdd[0] = f
        b_thdd[0] = xdd*c + self.G*s

        A = np.vstack((a_xdd, a_thdd))
        b = np.hstack((b_xdd, b_thdd))

        A_lp_1 = np.hstack((-A, -np.eye(2)))
        A_lp_2 = np.hstack((A, -np.eye(2)))

        A_lp = np.vstack((A_lp_1, A_lp_2))
        b_lp = np.hstack((-b, b))

        return (A_lp, b_lp)

    def update_set(self, obs, force):
        (A_new, b_new) = self.build_features(obs, force)
        (A_new, b_new) = self.add_slack(A_new, b_new)
        self.add_constraints(A_new, b_new)

    def add_slack(self, A, b):
        A_row_norms = np.sqrt(np.sum(A*A, 1)).flatten() #
        b += A_row_norms * self.EPS # adds EPS-slack in euclidian norm to LP constraints
        return (A, b)

    def add_constraints(self, A_new, b_new):
        newpoly_0 = pc.Polytope(A_new, b_new, normalize=True)
        (A_new, b_new) = newpoly_0.A, newpoly_0.b
        (A_old, b_old) = self.poly.A, self.poly.b
        A = np.vstack((A_old, A_new))
        b = np.hstack((b_old, b_new))
        newpoly = pc.Polytope(A, b)
        if newpoly.A.shape[0] > 200:
            # self.poly = pc.reduce(newpoly) ## polytope package is too slow to perform online reduction
            # TODO: use pycddlib
            self.poly = newpoly
        else:
            self.poly = newpoly

    def select_parameter(self, method='random'):
        dim = self.p_dim
        c = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # default choice just picks something feasible
        if method == 'random':
            c = self.np_rng.standard_normal(size=(dim,), dtype='float64')
            c[-2:] = abs(c[-2:]) # always choose smaller noise levels eta
        elif method == 'min_l1_eta':
            c = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        elif method == 'optimism_1':
            weight_th = 2
            c = np.array([-self.xdd_max, -self.xdd_max*weight_th, 0.0, 0.0, 0.0, 1.0, 1.0*weight_th])
        c_opt = matrix(c)
        A_opt = matrix(self.poly.A)
        b_opt = matrix(self.poly.b)
        sol = solvers.lp(c_opt, A_opt, b_opt, solvers='mosek')
        #res = lp(c, self.poly.A, self.poly.b)
        #assert res.success == True, "selecting parameter from LP failed"
        assert sol['status'] == 'optimal', "selecting parameter from LP failed"
        p_choice = np.array(sol['x']).flatten()
        return p_choice

    def select_parameter_stein(self, N=4):
        dim = self.p_dim
        c = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # default choice just picks something feasible
        p_samples = None
        for j in range(N):
            c = self.np_rng.standard_normal(size=(dim,), dtype='float64')
            c[-2:] = 20*abs(c[-2:]) # always choose smaller noise levels eta
            c_opt = matrix(c)
            A_opt = matrix(self.poly.A)
            b_opt = matrix(self.poly.b)
            sol = solvers.lp(c_opt, A_opt, b_opt, solvers='mosek')
            #res = lp(c, self.poly.A, self.poly.b)
            #assert res.success == True, "selecting parameter from LP failed"
            assert sol['status'] == 'optimal', "selecting parameter from LP failed"
            p_choice = np.array(sol['x']).flatten()
            if p_samples is None:
                p_samples = p_choice
            else:
                p_samples = np.vstack((p_samples, p_choice))

        p_final = np.mean(p_samples, axis=0)
        return p_final

    def tf_parameter(self, p):
        # transform [m_c + m_p, m_p * l, b_x, l, b_th, eta_x, eta_th] --> [m_c, m_p , b_x, l, b_th, eta_x, eta_th]
        (p1, p2, p3, p4, p5, eta1, eta2) = p
        m_p = p2/p4
        m_c = p1-m_p
        return np.array([m_c, m_p, p3, p4, p5, eta1, eta2])

if __name__ == '__main__':
    obs_log = np.loadtxt('../obs_log.csv')
    a_log = np.loadtxt('../a_log.csv')
    p_min = np.array([0.1, 0.01, 0, 0.01, 0, 0, 0])
    p_max = np.array([5, 1.1, 10, 2, 10, 40, 40])
    P = ConsistentPolytope(p_min, p_max, 0.02, 2)
    for (o, a) in zip(obs_log, a_log):
        f = 10*a
        P.update_set(o, f)
        print(np.shape(P.poly.A))
        #par = P.select_parameter(method='min_l1_eta')
        #par = P.select_parameter(method='random')
        par = P.select_parameter_stein()
        print("parameters: m_c = {}, m_p = {}, b_x = {}, l = {},"
              " b_th = {}, eta_x = {}, eta_th = {}".format(*(P.tf_parameter(par))))
