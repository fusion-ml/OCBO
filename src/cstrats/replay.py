"""
Optimization that replace trace of a previous run with a fixed GP.
"""

from cstrats.cts_opt import ContinuousOpt

class CtsReplay(ContinuousOpt):

    def set_fixed_gp(self, gp):
        """Set the GP to be used."""
        self.gp = gp

    def set_replay(self, replay):
        """Set the history that we are immitating.
        Args:
            replay: Namespace that is the history of a previous run.
        """
        self.replay = replay
        self.replay_qh = replay.query_history

    def set_clean(self):
        self.replay_counter = 0
        super(CtsReplay, self).set_clean()

    @staticmethod
    def get_strat_name():
        return 'replay'

    def _determine_next_query(self):
        if self.replay_qh is None or self.gp is None:
            raise RuntimeError('GP and replay need to be added.')
        pt = self.replay_qh[self.replay_counter].pt
        self.replay_counter += 1
        return pt

    def _child_set_up(self, function, domain, ctx_dim, options):
        self.pre_loaded_fit = True
        self.gp = None
        self.replay = None
        self.replay_qh = None
        self.replay_counter = 0

    def pre_load_points(self, eval_pts, rewards):
        """Give points to fit the GP, use this GP for the entire time.
        Args:
            eval_pts: List of lists representing points.
            rewards: List of rewards.
        """
        if self.gp is None:
            raise RuntimeError('GP needs to be set before any optimization.')
        for idx in range(len(rewards)):
            self.gp.add_data_single(eval_pts[idx], rewards[idx])
        self.x_init = eval_pts
        self.y_init = rewards
        self.pre_loaded_fit = True

    def _draw_next_gp(self):
        if self.gp is None:
            raise RuntimeError('GP needs to be set before any optimization.')
        else:
            return self.gp
