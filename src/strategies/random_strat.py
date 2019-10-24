"""
Strategy that picks which function and which point randomly.
"""
from __future__ import division

import numpy as np

from strategies.multi_opt import MultiOpt
from strategies.joint_opt import JointOpt

class RandomOpt(MultiOpt):

    def decide_next_query(self):
        """Choose function index and query point randomly."""
        idx = np.random.randint(len(self.fcns))
        low, high = zip(*self.domains[idx])
        pt = np.random.uniform(low, high)
        return idx, pt

    @staticmethod
    def get_opt_method_name():
        return 'Random'

    def _draw_next_gps(self):
        if self.risk_neutral:
            return super(RandomOpt, self)._draw_next_gps()
        else:
            pass

    def _update_models(self):
        if self.risk_neutral:
            return super(RandomOpt, self)._update_models()
        else:
            pass

    def _add_point_to_gp(self, f_idx, pt, val):
        if self.risk_neutral:
            return super(RandomOpt, self)._add_point_to_gp(f_idx, pt, val)
        else:
            pass

class JointRandom(JointOpt):

    def decide_next_query(self):
        """Choose function index and query point randomly."""
        idx = np.random.randint(len(self.fcns))
        low, high = zip(*self.domains[idx])
        pt = np.random.uniform(low, high)
        return idx, pt

    @staticmethod
    def get_opt_method_name():
        return 'joint-rand'

    def _draw_next_gps(self):
        if self.risk_neutral:
            return super(JointRandom, self)._draw_next_gps()
        else:
            pass

    def _update_models(self):
        if self.risk_neutral:
            return super(JointRandom, self)._update_models()
        else:
            pass

    def _add_point_to_gp(self, f_idx, pt, val):
        if self.risk_neutral:
            return super(JointRandom, self)._add_point_to_gp(f_idx, pt, val)
        else:
            pass
