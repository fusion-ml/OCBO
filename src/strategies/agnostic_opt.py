"""
Class of strategies that are agnostic to context and picks uniformly at random.
"""

from argparse import Namespace
import numpy as np

from strategies.multi_opt import MultiOpt
from util.misc_util import thomp_acq, ei_acq, build_gp_posterior

class AgnosticOpt(MultiOpt):

    def _agn_child_decide_query(self, idx):
        """Given a particular gp, decide best point to query.
        Args:
            idx: Index of GP and domain to use.
        Returns: Point on GP to query.
        """
        raise NotImplementedError('To be implemented in child.')

    def decide_next_query(self):
        """Make a query that decides which function and which point.
        Returns: Index of functions selected and point to query.
        """
        idx = np.random.randint(len(self.fcns))
        build_gp_posterior(self.gps[idx])
        query_pt = self._agn_child_decide_query(idx)
        return idx, query_pt

class AgnEI(AgnosticOpt):

    def _agn_child_decide_query(self, idx):
        """Given a particular gp, decide best point to query.
        Args:
            idx: Index of GP and domain to use.
        Returns: Point on GP to query.
        """
        gp = self.gps[idx]
        domain = self.domains[idx]
        curr_max_val = self.curr_best[self.f_names[idx]][-1][0]
        return ei_acq(gp, domain, self.options.max_opt_evals, curr_max_val)[0]

    @staticmethod
    def get_opt_method_name():
        """Get type of agnostic method as string."""
        return 'agn-ei'

class AgnThompson(AgnosticOpt):

    def _agn_child_decide_query(self, idx):
        """Given a particular gp, decide best point to query.
        Args:
            idx: Index of GP and domain to use.
        Returns: Point on GP to query.
        """
        gp = self.gps[idx]
        domain = self.domains[idx]
        curr_max_val = self.curr_best[self.f_names[idx]][-1][0]
        return thomp_acq(gp, domain, self.options.max_opt_evals,
                         curr_max_val)[0]

    @staticmethod
    def get_opt_method_name():
        """Get type of agnostic method as string."""
        return 'agn-thomp'

agn_strats = [Namespace(name=AgnEI.get_opt_method_name(), impl=AgnEI),
              Namespace(name=AgnThompson.get_opt_method_name(),
                        impl=AgnThompson)
]
