import numpy as np
from message_types.msg_state import MsgState
import parameters.simulation_parameters as SIM


class Guidance:
    def __init__(self):
        foo = 0.  # do nothing at initialization

    def update(self, ownship_state, bearings, sizes):
        print("bearing: ", bearings)
        print("pixel sizes: ", sizes)
        u = 0.  # no guidance
        return u
