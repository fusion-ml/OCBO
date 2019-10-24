from synth.function_generator import add_funcs
from synth.oned import oneD_functions
from synth.twod import twod_functions
from synth.sixd import sixd_functions

synth_functions = oneD_functions + twod_functions + sixd_functions + add_funcs
