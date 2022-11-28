import numpy as np
import math
from .clc_candidate_policy import candidate
from .consistent_set import ConsistentPolytope

class clc:
    KEY = ('m_c', 'm_p', 'b_x', 'l', 'b_th')
    """
    p = [m_c + m_p, m_p * l, b_x, l, b_th, eta_x, eta_th]
    
    par = [m_c, m_p , b_x, l, b_th, eta_x, eta_th]
    """
    def __init__(self):
        p_min = np.array([1.1, 0.01, 0, 0.01, 0, 0, 0])
        p_max = np.array([5, 3, 10, 2, 10, 10, 10])
        self.consistent_set = ConsistentPolytope(p_min, p_max, 0.02, 2)
        self.sample_method = 'optimism_1'
        #self.sample_method = 'steiner'
        self.scale_action = 10
        self.fmax = 200
        self.policy, self.p = self.sample_candidate()
        self.obs_f_log = None

    def sample_candidate(self):
        p = None
        if self.sample_method == 'steiner':
            p = self.consistent_set.select_parameter_stein()
        else:
            p = self.consistent_set.select_parameter(self.sample_method)
        par = self.consistent_set.tf_parameter(p)
        (m_c, m_p, b_x, l, b_th, _, _) = par
        cand_idx = dict(zip(self.KEY, (m_c, m_p, b_x, l, b_th)))
        return (candidate(cand_idx), p)

    def get_action(self, obs, reset=False, verbose=False):
        action = self.policy.compute_action(obs)
        if self.obs_f_log is None:
            self.obs_f_log = [[[], action[0]*self.scale_action]]
        elif reset:
            self.obs_f_log[-1] = [[], action[0]*self.scale_action]
        else:
            self.obs_f_log[-1][0] = obs
            prev_force = self.obs_f_log[-1][1]
            self.consistent_set.update_set(obs, prev_force)
            self.obs_f_log.append([[], action[0]*self.scale_action])
            if not (self.p in self.consistent_set.poly):
                self.policy, self.p = self.sample_candidate()
                if verbose:
                    print("old p inconsistent, new p: {}".format(self.p))
        return action