from argparse import Namespace

from strategies.agnostic_opt import agn_strats
from strategies.corr_opt import corr_strats, corr_args
from strategies.multi_opt import multi_opt_args
from strategies.random_strat import RandomOpt, JointRandom
from strategies.mei import MEI
from strategies.mts import MTS
from strategies.joint_agnostic_opt import ja_strats
from strategies.joint_mei import JointMEI
from strategies.joint_mts import JointMTS

discrete_strategies = [Namespace(name='Random', impl=RandomOpt),
                       Namespace(impl=MTS, name=MTS.get_opt_method_name()),
                       Namespace(impl=MEI, name=MEI.get_opt_method_name())] \
                      + agn_strats \
                      + corr_strats

joint_strategies = [Namespace(impl=JointRandom,
                              name=JointRandom.get_opt_method_name()),
                    Namespace(impl=JointMTS,
                              name=JointMTS.get_opt_method_name()),
                    Namespace(impl=JointMEI,
                              name=JointMEI.get_opt_method_name())] \
                   + ja_strats

strategies = discrete_strategies + joint_strategies

strat_args =  corr_args + multi_opt_args
