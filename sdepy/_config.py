"""
==============================
PRIVATE CONFIGURATION SETTINGS
==============================

Global settings for:
*  ``kfunc`` deployment,
*  unit testing behaviour.
"""

# kfunc deployment (see .shortcuts)
# ---------------------------------
KFUNC = 'shortcuts'


# testing mode
# ------------

# if False, quantitative tests do not execute plot construction;
# if True, needs matplotlib.pyplot to be installed
PLOT = False

# if OUTPUT_DIR is not None, quantitative tests save
# in this folder the realized errors and, if generated,
# plot images.
OUTPUT_DIR = None

VERBOSE = False

# random number generator, or function returning a
# properly seeded such generator, to be used in tests;
# if set to 'legacy', use numpy legacy random number generation,
# and check expected vs. realized errors, failing if found
# inconsistent
TEST_RNG = 'legacy'

# number of paths generated while executing quantitative tests;
# .tests.test_montecarlo quant tests cumulate 100*PATHS paths
# .tests.test_quant geterates processes with PATH paths,
# using order of 100 steps (the actual number of steps depends on the process)
PATHS = 100
